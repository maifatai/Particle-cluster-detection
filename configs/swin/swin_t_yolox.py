_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
# _base_ = ['../_base_/default_runtime.py']
max_epochs = 10000
batch_size=24 #batch size=12,memory=32
img_scale = (512, 512)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,#注意是embed_dim，不是embed_dims
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),#需要修改 Output from which stages.
        # with_cp=False,#没有之后的这三个参数
        # convert_weights=True,
        # init_cfg=None
        ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,# 金字塔特征图每一层的输出通道
        num_csp_blocks=1#CSPblock的数量
        ),
    bbox_head=dict(
        type='YOLOXHead', num_classes=2, in_channels=192, feat_channels=192),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
data_root = 'data/cluster/'
dataset_type = 'CocoDataset'





data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='data/cluster/annotations/instances_train2017.json',
            img_prefix='data/cluster/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-256, -256)),
            dict(
                type='MixUp',
                img_scale=(512, 512),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/cluster/annotations/instances_val2017.json',
        img_prefix='data/cluster/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/cluster/annotations/instances_val2017.json',
        img_prefix='data/cluster/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

# optimizer
# default 8 gpu
# optimizer = dict(
#     # _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         })
#         )
optimizer = dict(# 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与 PyTorch 里的优化器参数一致。
    type='SGD',# 优化器种类，更多细节可参考 mmdet/core/optimizer/default_constructor.py#L13
    lr=0.00125,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))



optimizer_config = dict(#optimizer hook 的配置文件  mmcv/runner/hooks/optimizer.py
    grad_clip=None) # 大多数方法不使用梯度限制(grad_clip)。



num_last_epochs = 15
resume_from = None
interval = 1

# learning policy
lr_config = dict(# 学习率调整配置，用于注册 LrUpdater hook。
    _delete_=True,
    policy='YOLOX',# 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等
    warmup='exp', # 预热(warmup)策略，也支持 `exp` 和 `constant`,'linear'。
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,# 用于热身的起始学习率的比率
    warmup_iters=5,  # 5 epoch# 预热的迭代次数
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(
    type='EpochBasedRunner', # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
    max_epochs=max_epochs)# runner 总回合数， 对于 IterBasedRunner 使用 `max_iters`

checkpoint_config = dict(# Checkpoint hook 的配置文件
    interval=50)# 保存的间隔。


log_config = dict(# register logger hook 的配置文件。
    interval=50,# 打印日志的间隔
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
    ]
    )
evaluation=dict( # evaluation hook 的配置
    interval=1,# 验证的间隔。
    metric=['bbox'],# 验证期间使用的指标。
    save_best='bbox_mAP') 

# 需要使用 _delete_=True 将新的键去替换 backbone 域内所有老的键