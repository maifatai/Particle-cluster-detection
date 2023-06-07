_base_=["./convnext_t_rcnn.py"]
batch_size=2    #batch_size=2 32G   batch_size=8 mem=80
data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=2)
model=dict(
    backbone=dict(
    depths=[3,3,27,3],
    dims=[192,384,768,1536],
    drop_path_rate=0.3
    ),
    neck=dict(
        in_channels=[192,384,768,1536]
    )  
)