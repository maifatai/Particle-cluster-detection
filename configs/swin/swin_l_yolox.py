_base_='./swin_t_yolox.py'

model=dict(
    backbone=dict(
        depths=[2,2,18,2],
        embed_dim=192,
        num_heads=[6, 12, 24, 48],
    ),
    neck=dict(
        in_channels=[384,768,1536],
        out_channels=384,
        num_csp_blocks=2
    ),
    bbox_head=dict(
        in_channels=384,
        feat_channels=384,
    )
)
batch_size=12#batch_size=3,mem=32
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)
checkpoint_config = dict(# Checkpoint hook 的配置文件
    interval=50)# 保存的间隔