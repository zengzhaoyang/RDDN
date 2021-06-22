# RDDN

## Introduction

This is the code for paper ``Reference-based Defect Detection Network''. 

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Notes

This repo is developed based on MMDetection. The main modified files are

* mmdet/datasets/pipelines/loading.py
* mmdet/models/detectors/two_stage.py
* mmdet/models/detectors test_mixins.py
* mmdet/models/detectors/faster_rcnn.py
* mmdet/models/detectors/cascade_rcnn.py

Since we do not own the copyright of the dataset images, the dataset images should be downloaded from [https://tianchi.aliyun.com/competition/entrance/231682/information](https://tianchi.aliyun.com/competition/entrance/231682/information) and [https://tianchi.aliyun.com/competition/entrance/231748/information](https://tianchi.aliyun.com/competition/entrance/231748/information). 
