_base_=["./swin_t_rcnn.py"]
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.2,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

batch_size=16 #batch_size=48,GPU memeory=32
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)