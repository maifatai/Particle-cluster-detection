_base_='./swin_t_yolox.py'

model=dict(
    backbone=dict(
        depths=[2,2,18,2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
    ),
    neck=dict(
        in_channels=[256,512,1024],
        out_channels=256,
        num_csp_blocks=2
    ),
    bbox_head=dict(
        in_channels=256,
        feat_channels=256,
    )
)
batch_size=16 #batch_size=6,mem=32
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)