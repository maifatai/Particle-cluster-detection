checkpoint_config = dict( #Checkpoint hook 的配置文件
    interval=200)
# yapf:disable
log_config = dict(# register logger hook 的配置文件。
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')# 同样支持 Tensorboard 日志
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl') # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'# 日志的级别。
load_from = None# 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None# 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]# runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。根据 total_epochs 工作流训练 12个回合。
# work_dir = 'work_dir'  # 用于保存当前实验的模型检查点和日志的目录。