# Cluster-Locating Algorithm Based on Deep Learning for Silicon Pixel Sensors

This repo contains the supported code and configuration files, from the following paper:

[Cluster-Locating Algorithm Based on Deep Learning for Silicon Pixel Sensors](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).


## Results and Models


### One-stage model

   | Backbone | AP $(\%)$ | AP $_{50}(\%)$ | AP $_{75}(\%)$ | FPS | Params.(M) | FLOPs(G) | model |
   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
   | ConvNeXt-T-YOLOX | 49.2 | 85.0 | 54.0 | 57.94 | 38.44 |  8.64 |  |
   | Swin-T-YOLOX | 57.2 | 90.0 | 66.7 | 41.47 | 38.12 | 9.93 | |
   | ConvNeXt-S-YOLOX | 48.1 | 84.0 | 50.4 | 42.40 | 62.3 4| 14.55 | |
  | Swin-S-YOLOX | 54.9 | 87.2 | 64.1 | 27.93 | 61.71 | 17.42 | |
  | ConvNeXt-B-YOLOX | 48.8 | 83.9 | 53.6 | 30.96 | 110.48 | 25.74 | |
  | Swin-B-YOLOX | 55.1  |87.7| 64.9|27.57 | 109.65| 30.93 | |
  | ConvNeXt-L-YOLOX | 48.9 |84.4|51.0|17.04| 247.80| 57.67 | |
  | Swin-L-YOLOX  | 55.6 |87.1|65.5|15.64| 246.56| 69.52 | |
  
### One-stage model

   | Backbone | AP $(\%)$ | AP $_{50}(\%)$ | AP $_{75}(\%)$ | FPS | Params.(M) | FLOPs(G) | model |
   | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
    Swin-T-RCNN | 64.3|90.8|82.6|25.18  | 44.72| 27.65 |  |
    ConvNeXt-T-RCNN | 66.4 |91.9|83.9|27.24  |45.03| 26.36 |  |
    
    Swin-S-RCNN | 62.6 |89.4|80.1|18.58  | 66.02| 34.76 |  |
    ConvNeXt-S-RCNN | 66.5 |93.3|82.3 |20.41  |66.64| 31.88 |  |
    
    Swin-B-RCNN | 62.7 |90.1|81.1|14.66  | 104.03| 45.85 |  |
    ConvNeXt-B-RCNN | 65.4  |90.3|82.0|16.31  |104.86| 40.65 |  |
    
    Swin-L-RCNN | 66.7 |92.8|84.8|9.34 | 212.50| 77.46 |  |
    ConvNeXt-L-RCNN  | 68.0 |93.4|86.2 |10.04  | 213.74|65.61 |  |


   
| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 43.7 | 39.8 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1bYZk7BIeFEozjRNUesxVWg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/19UOW0xl0qc-pXQ59aFKU5w) |
| Swin-T | ImageNet-1K | 3x | 46.0 | 41.6 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1Te-Ovk4yaavmE4jcIOPAaw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1YpauXYAFOohyMi3Vkb6DBg) |
| Swin-S | ImageNet-1K | 3x | 48.5 | 43.3 | 69M | 359G | [config](configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1ymCK7378QS91yWlxHMf1yw) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1V4w4aaV7HSjXNFTOSA6v6w) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1x4vnorYZfISr-d_VUSVQCA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1vFwbN1iamrtwnQSxMIW4BA) |
| Swin-T | ImageNet-1K | 3x | 50.4 | 43.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1GW_ic617Ak_NpRayOqPSOA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1i-izBrODgQmMwTv6F6-x3A) |
| Swin-S | ImageNet-1K | 3x | 51.9 | 45.0 | 107M | 838G | [config](configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/17Vyufk85vyocxrBT1AbavQ) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1Sv9-gP1Qpl6SGOF6DBhUbw) |
| Swin-B | ImageNet-1K | 3x | 51.9 | 45.0 | 145M | 982G | [config](configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1UZAR39g-0kE_aGrINwfVHg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1tHoC9PMVnldQUAfcF6FT3A) |

### RepPoints V2

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | ImageNet-1K | 3x | 50.0 | - | 45M | 283G |

### Mask RepPoints V2

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Swin-T | ImageNet-1K | 3x | 50.3 | 43.6 | 47M | 292G |

**Notes**: 

- **Pre-trained models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.

## Results of MoBY with Swin Transformer

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 43.6 | 39.6 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1P5gCIfLUQ64jbVMOom0H3w) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1xGRihuIrGVreFKn5eJ6oTg) |
| Swin-T | ImageNet-1K | 3x | 46.0 | 41.7 | 48M | 267G | [config](configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_3x.log.json)/[baidu](https://pan.baidu.com/s/17WAhUmhAam1of3hXOu-wtA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_mask_rcnn_swin_tiny_patch4_window7_3x.pth)/[baidu](https://pan.baidu.com/s/1MSj8cC1wlQU1QaXCdKrzeA) |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.5 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1eOdq1rvi0QoXjc7COgiM7A) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1-gbY-LExbf0FgYxWWs8OPg) |
| Swin-T | ImageNet-1K | 3x | 50.2 | 43.5 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.log.json)/[baidu](https://pan.baidu.com/s/1zEFXHYjEiXUCWF1U7HR5Zg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/moby_cascade_mask_rcnn_swin_tiny_patch4_window7_3x.pth)/[baidu](https://pan.baidu.com/s/1FMmW0GOpT4MKsKUrkJRgeg) |

**Notes:**

- The drop path rate needs to be tuned for best practice.
- MoBY pre-trained models can be downloaded from [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

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



