_base_=["./convnext_t_rcnn.py"]

batch_size=6  #batchsize=6 32   batch=16 mem=80
data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=2)
model=dict(
    backbone=dict(
    depths=[3,3,27,3],
    dims=[96,192,384,768],
    drop_path_rate=0.2
    ),
    neck=dict(
        in_channels=[96,192,384,768]
    )  
)
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )