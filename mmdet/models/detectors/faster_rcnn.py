from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler

import torch
import torch.nn as nn
import torch.nn.functional as F

@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 context_reference=-1):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            context_reference=context_reference)



class Generator(nn.Module):

    def __init__(self, in_channel):
        super(Generator, self).__init__()

        #self.conv_proj = nn.Sequential(nn.Conv2d(in_channel, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv_proj = nn.Conv2d(in_channel, 256, kernel_size=1)

        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)

        self.conv3_rev = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv2_rev = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.conv1_rev = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        #x = self.conv_proj(x)
        #x = F.relu(x, inplace=True)
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        shape1 = x.shape[2:]
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        shape2 = x.shape[2:]
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv3_rev(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, shape2)
        x = self.conv2_rev(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, shape1)
        x = self.conv1_rev(x)
        #x = F.relu(x, inplace=True)
        return x 


@DETECTORS.register_module
class FasterRCNN_PTG(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 context_reference=-1):
        super(FasterRCNN_PTG, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            context_reference=context_reference)

        in_channels = [256, 512, 1024, 2048]
        #self.G = [Generator(i) for i in in_channels]
        self.G0 = Generator(256)
        self.G1 = Generator(512)
        self.G2 = Generator(1024)
        self.G3 = Generator(2048)

        self.crit_mse = nn.MSELoss()

    def forward_train(self, img,
                            img_metas,
                            gt_bboxes,
                            gt_labels,
                            gt_bboxes_ignore=None,
                            gt_masks=None,
                            proposals=None,
                            normal=None):

        device = img.device
        x = self.backbone(img)

        with torch.no_grad():
            xn = self.backbone(normal)


        xn = [self.G0.conv_proj(xn[0]), self.G1.conv_proj(xn[1]), self.G2.conv_proj(xn[2]), self.G3.conv_proj(xn[3])]
        xn_g = [self.G0(xn[0]), self.G1(xn[1]), self.G2(xn[2]), self.G3(xn[3])]

        #xg = [g(i) for g, i in zip(self.G, x)]
        xp = [self.G0.conv_proj(x[0]), self.G1.conv_proj(x[1]), self.G2.conv_proj(x[2]), self.G3.conv_proj(x[3])]
        xp_g = [self.G0(xp[0]), self.G1(xp[1]), self.G2(xp[2]), self.G3(xp[3])]

      
        loss_generate = sum([self.crit_mse(xn[i], xn_g[i]) for i in range(4)])

        x = [i - j for i, j in zip(xp, xp_g)]
        x = self.neck(x)

        losses = dict()
        losses['loss_generate'] = loss_generate

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)


        else:
            proposal_list = proposals


        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])

                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)


        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])

            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            if self.context_reference > 0:
                rois_center = (rois[:, 1:3] + rois[:, 3:5]) / 2
                rois_wh = (rois[:, 3:5] - rois[:, 1:3]) * self.context_reference
                rois_context = rois.clone()
                rois_context[:, 1:3] = (rois_center - rois_wh/2).clamp(min=0)
                rois_context[:, 3:5] = (rois_center + rois_wh/2)
                h, w = img_metas[0]['pad_shape'][:2]
                rois_context[:, 3] = rois_context[:, 3].clamp(max=w)
                rois_context[:, 4] = rois_context[:, 4].clamp(max=h)

                context_feats = self.bbox_roi_extractor(
                        x[:self.bbox_roi_extractor.num_inputs], rois_context)
                bbox_feats = torch.cat((bbox_feats, context_feats), dim=1)

            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            else:
                cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.backbone(img)

        xp = [self.G0.conv_proj(x[0]), self.G1.conv_proj(x[1]), self.G2.conv_proj(x[2]), self.G3.conv_proj(x[3])]
        xg = [self.G0(xp[0]), self.G1(xp[1]), self.G2(xp[2]), self.G3(xp[3])]

        x = [i - j for i, j in zip(xp, xg)]
        x = self.neck(x)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)


        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results
