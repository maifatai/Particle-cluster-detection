_base_=['./convnext_t_yolox.py']

model=dict(
    backbone=dict(
      depths=[3, 3, 27, 3],
      dims=[192,384,768,1536] ,
      drop_path_rate=0.2 
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
batch_size=2#batch_size=2,mem=32  batch_size=6 mem=80

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2
    )


