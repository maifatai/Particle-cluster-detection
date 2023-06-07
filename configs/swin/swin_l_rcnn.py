_base_=["./swin_t_rcnn.py"]
model = dict(
    backbone=dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.4,
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]))

batch_size=8#batch_size=32,GPU memeory=32G
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)