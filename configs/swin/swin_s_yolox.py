_base_='./swin_t_yolox.py'

model=dict(
    backbone=dict(
        depths=[2,2,18,2]
    ),
    neck=dict(
        num_csp_blocks=2
    )
)
batch_size=16 #batchsize=6 mem=32
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)

