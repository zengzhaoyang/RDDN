import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

import zipfile
import cv2

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)

class ZipReader(object):
    def __init__(self, fname):
        super(ZipReader, self).__init__()
        self.id_context = dict()
        self.fname = fname

    def read(self, image_name):
        if self.fname in self.id_context:
            return self.id_context[self.fname].read(image_name)
        else:
            file_handle = zipfile.ZipFile(self.fname, 'r')
            self.id_context[self.fname] = file_handle
            return self.id_context[self.fname].read(image_name)

    def namelist(self):
        if self.fname in self.id_context:
            return self.id_context[self.fname].namelist()
        else:
            file_handle = zipfile.ZipFile(self.fname, 'r')
            self.id_context[self.fname] = file_handle
            return self.id_context[self.fname].namelist()

@PIPELINES.register_module
class LoadImageFromZip(object):

    def __init__(self, fname, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.z = ZipReader(fname)

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = results['img_prefix'] + results['img_info']['filename']
        else:
            filename = results['img_info']['filename']

        #img = mmcv.imread(filename, self.color_type)
        data = self.z.read(filename)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadImageFromZipTRPre(object):

    def __init__(self, fname, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.z = ZipReader(fname)

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = results['img_prefix'] + results['img_info']['filename']
        else:
            filename = results['img_info']['filename']

        #img = mmcv.imread(filename, self.color_type)
        data = self.z.read(filename)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        tfilename = filename.replace('defect', 'template')
        tmp = tfilename.split('/')
        tmp[-1] = 'template_' + tmp[-1].split('_')[0] + '.jpg'
        tfilename = '/'.join(tmp)
        data = self.z.read(tfilename)
        timg = cv2.imdecode(np.frombuffer(data, np.uint8), 1).astype(np.float32)

        img = img - timg

        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)

@PIPELINES.register_module
class LoadImageFromZipTRInter(object):

    def __init__(self, fname, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.z = ZipReader(fname)

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = results['img_prefix'] + results['img_info']['filename']
        else:
            filename = results['img_info']['filename']

        #img = mmcv.imread(filename, self.color_type)
        data = self.z.read(filename)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        tfilename = filename.replace('defect', 'template')
        tmp = tfilename.split('/')
        tmp[-1] = 'template_' + tmp[-1].split('_')[0] + '.jpg'
        tfilename = '/'.join(tmp)
        data = self.z.read(tfilename)
        timg = cv2.imdecode(np.frombuffer(data, np.uint8), 1).astype(np.float32)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = img
        results['template'] = timg
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadImageWithNormalFromZip(object):

    def __init__(self, fname, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.z = ZipReader(fname)
        self.normal_list = []
        self.normal_num = 0

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = results['img_prefix'] + results['img_info']['filename']
        else:
            filename = results['img_info']['filename']

        #img = mmcv.imread(filename, self.color_type)
        data = self.z.read(filename)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        if self.normal_num == 0:
            tmp = self.z.namelist()
            for item in tmp:
                if 'normal' in item and '.jpg' in item:
                    self.normal_list.append(item)
            self.normal_num = len(self.normal_list)


        nid = np.random.randint(self.normal_num)
        nimg = self.z.read(self.normal_list[nid])
        nimg = cv2.imdecode(np.frombuffer(nimg, np.uint8), 1)


        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['normal'] = nimg
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadMultiChannelImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.
    Expects results['filename'] to be a list of filenames
    """

    def __init__(self, to_float32=True, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        if len(results['gt_bboxes']) == 0:
            print(results)

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
