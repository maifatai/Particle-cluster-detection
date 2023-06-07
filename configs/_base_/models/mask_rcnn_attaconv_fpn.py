# model settings
model = dict(
    type='MaskRCNN',#位于/mmbed/models/detectors/mask_rcnn.py的MaskRCNN类
    pretrained=None,#MaskRCNN类的参数
    backbone=dict(#对应mmdet/models/backbones/文件夹中的文件
        type='AttentionConv',
        patch_size=4,
        in_channels=3,
        dim=96, 
        num_heads=[3,6,12,24],
        depths=[2,2,6,2], 
        out_indices=(0,1,2,3), 
        attn_drop=0.2,
        proj_drop=0.2,
        drop_path=0.2,
        conv_scale=4,
        ),
    neck=dict(#对应mmdet/models/necks/文件夹中的文件
        type='FPN',#对应mmdet/models/necks/fpn.py 的FPN 类
        in_channels=[96, 192, 384, 768],#输入参数 # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,# 金字塔特征图每一层的输出通道
        num_outs=5),# 输出的范围(scales)
    rpn_head=dict(
        type='RPNHead',#对应mmdet/models/dense_head/rpn_head.py 中的RPNHead类
        in_channels=256,# 每个输入特征图的输入通道，这与 neck 的输出通道一致。
        feat_channels=256,# head 卷积层的特征通道。
        anchor_generator=dict(#mmdet/core/anchor/文件夹
            type='AnchorGenerator',#mmdet/core/anchor/anchor_generator.py中的AnchorGenerator类
            scales=[8],#AnchorGenerator类的参数 # 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0],# 高度和宽度之间的比率。
            strides=[4, 8, 16, 32, 64]),# 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',#mmdet/core/bbox/coder/delta_xywh_bbox_coder.py中的DeltaXYWHBBoxCoder类
            target_means=[.0, .0, .0, .0],#DeltaXYWHBBoxCoder类的参数
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),#对应mmdet/models/losses/cross_entropy_loss.py中的CrossEntropyLoss类及参数
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(#对应mmdet/models/roi_head/文件夹中的文件
        type='StandardRoIHead',#对应mmdet/models/roi_head/standard_roi_head.py 中的StandardRoIHead类
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',#对应#对应mmdet/models/roi_head/roi_heads/single_level_roi_extractor.py的SingleRoIExtractor类
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),#SingleRoIExtractor类的输入
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',#对应mmdet/models/roi_head/bbox_heads/convfc_bbox_head.py的Shared2FCBBoxHead类
            in_channels=256,#Shared2FCBBoxHead类的输入
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',#mmdet/core/bbox/coder/delta_xywh_bbox_coder.py中的DeltaXYWHBBoxCoder类
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        # mask_roi_extractor=dict(#
        #     type='SingleRoIExtractor',#对应#对应mmdet/models/roi_head/roi_heads/single_level_roi_extractor.py的SingleRoIExtractor类
        #     roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        # mask_head=dict(#对应mmdet/models/roi_head/mask_heads/文件夹
        #     type='FCNMaskHead',#对应mmdet/models/roi_head/mask_heads/fcn_mask_head.py中的FCNMaskHead类
        #     num_convs=4,#FCNMaskHead类的输入
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=2,
        #     loss_mask=dict(
        #         type='CrossEntropyLoss',#对应mmdet/models/losses/cross_entropy_loss.py中的CrossEntropyLoss类
        #         use_mask=True,# CrossEntropyLoss类的输入
        #         loss_weight=1.0))
                    mask_roi_extractor=None,
                    mask_head=None
                ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
