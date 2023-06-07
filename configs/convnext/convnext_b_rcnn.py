_base_=["./convnext_t_rcnn.py"]

batch_size=4   #batch_size=4  32G   batch_size=12 mem=80
data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=2)
model=dict(
    backbone=dict(
    depths=[3,3,27,3],
    dims=[128,256,512,1024],
    drop_path_rate=0.3
    ),
    neck=dict(
       in_channels=[128,256,512,1024]
    )  
)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)