_base_=["./swin_t_rcnn.py"]
model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]))

batch_size=16#batch_size=32,GPU memeory=32
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)