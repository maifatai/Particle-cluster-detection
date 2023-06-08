# Cluster-Locating Algorithm Based on Deep Learning for Silicon Pixel Sensors

This repo contains the supported code and configuration files, from the following paper:

[Cluster-Locating Algorithm Based on Deep Learning for Silicon Pixel Sensors](https://www.mdpi.com/1424-8220/23/9/4383). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).


## Results and Models


### One-stage model

   | Backbone | AP $(%)$ | AP $_{50}(%)$ | AP $_{75}(%)$ | FPS | Params.(M) | FLOPs(G) | model |
   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
   | ConvNeXt-T-YOLOX | 49.2 | 85.0 | 54.0 | 57.94 | 38.44 |  8.64 |  |
   | Swin-T-YOLOX | 57.2 | 90.0 | 66.7 | 41.47 | 38.12 | 9.93 | |
   | ConvNeXt-S-YOLOX | 48.1 | 84.0 | 50.4 | 42.40 | 62.3 4| 14.55 | |
  | Swin-S-YOLOX | 54.9 | 87.2 | 64.1 | 27.93 | 61.71 | 17.42 | |
  | ConvNeXt-B-YOLOX | 48.8 | 83.9 | 53.6 | 30.96 | 110.48 | 25.74 | |
  | Swin-B-YOLOX | 55.1  |87.7| 64.9|27.57 | 109.65| 30.93 | |
  | ConvNeXt-L-YOLOX | 48.9 |84.4|51.0|17.04| 247.80| 57.67 | |
  | Swin-L-YOLOX  | 55.6 |87.1|65.5|15.64| 246.56| 69.52 | [github](https://github.com/maifatai/Particle-cluster-detection/releases/download/v1.0/swin_l_yolox.pth)|
  
### Two-stage model

  | Backbone | AP $(%)$ | AP $_{50}(%)$ | AP $_{75}(%)$ | FPS | Params.(M) | FLOPs(G) | model |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
  |  Swin-T-RCNN | 64.3|90.8|82.6|25.18  | 44.72| 27.65 |  |
  |  ConvNeXt-T-RCNN | 66.4 |91.9|83.9|27.24  |45.03| 26.36 |  |
  |  Swin-S-RCNN | 62.6 |89.4|80.1|18.58  | 66.02| 34.76 |  |
  |  ConvNeXt-S-RCNN | 66.5 |93.3|82.3 |20.41  |66.64| 31.88 |  | 
  | Swin-B-RCNN | 62.7 |90.1|81.1|14.66  | 104.03| 45.85 |  |
  | ConvNeXt-B-RCNN | 65.4  |90.3|82.0|16.31  |104.86| 40.65 |  | 
  |  Swin-L-RCNN | 66.7 |92.8|84.8|9.34 | 212.50| 77.46 |  |
  |  ConvNeXt-L-RCNN  | 68.0 |93.4|86.2 |10.04| 213.74|65.61 | [github](https://github.com/maifatai/Particle-cluster-detection/releases/download/v1.0/convnext_l_rcnn.zip.001) [github](https://github.com/maifatai/Particle-cluster-detection/releases/download/v1.0/convnext_l_rcnn.zip.002) |



## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation.

### Training

```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example:
```
python -m torch.distributed.launch --nproc_per_node=1  tools/train.py configs/convnext/conv_l_rcnn.py --cfg-options data.samples_per_gpu=8 

```

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox 

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox 
```



###  Testing

Refer to the demo/one-stage.ipynb and demo/two-stage.ipynb files.






## Citing 
```
@Article{s23094383,
AUTHOR = {Mai, Fatai and Yang, Haibo and Wang, Dong and Chen, Gang and Gao, Ruxin and Chen, Xurong and Zhao, Chengxin},
TITLE = {Cluster-Locating Algorithm Based on Deep Learning for Silicon Pixel Sensors},
JOURNAL = {Sensors},
VOLUME = {23},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {4383},
URL = {https://www.mdpi.com/1424-8220/23/9/4383},
PubMedID = {37177585},
ISSN = {1424-8220},
DOI = {10.3390/s23094383}
}
```



