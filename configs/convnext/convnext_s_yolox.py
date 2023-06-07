_base_=['./convnext_t_yolox.py']

model=dict(
    backbone=dict(
      depths=[3, 3, 27, 3]  
    ),
    neck=dict(
        num_csp_blocks=2
    )
)
batch_size=6#batch_size=6,mem=32 batch=16 mem=80
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)