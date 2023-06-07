_base_=['./convnext_t_yolox.py']

model=dict(
    backbone=dict(
      depths=[3, 3, 27, 3],
      dims=[128,256,512,1024] ,
      drop_path_rate=0.2
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
batch_size=4#batch_size=4,mem=32  batchsize=12 mem=80G
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2)