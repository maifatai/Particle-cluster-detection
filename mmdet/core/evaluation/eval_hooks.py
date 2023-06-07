# import os.path as osp
# import warnings
# from math import inf

# import mmcv
# import torch.distributed as dist
# from mmcv.runner import Hook
# from torch.nn.modules.batchnorm import _BatchNorm
# from torch.utils.data import DataLoader

# from mmdet.utils import get_root_logger


# class EvalHook(Hook):
#     """Evaluation hook.

#     Notes:
#         If new arguments are added for EvalHook, tools/test.py,
#         tools/analysis_tools/eval_metric.py may be effected.

#     Attributes:
#         dataloader (DataLoader): A PyTorch dataloader.
#         start (int, optional): Evaluation starting epoch. It enables evaluation
#             before the training starts if ``start`` <= the resuming epoch.
#             If None, whether to evaluate is merely decided by ``interval``.
#             Default: None.
#         interval (int): Evaluation interval (by epochs). Default: 1.
#         save_best (str, optional): If a metric is specified, it would measure
#             the best checkpoint during evaluation. The information about best
#             checkpoint would be save in best.json.
#             Options are the evaluation metrics to the test dataset. e.g.,
#             ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
#             segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
#             ``auto``, the first key will be used. The interval of
#             ``CheckpointHook`` should device EvalHook. Default: None.
#         rule (str, optional): Comparison rule for best score. If set to None,
#             it will infer a reasonable rule. Keys such as 'mAP' or 'AR' will
#             be inferred by 'greater' rule. Keys contain 'loss' will be inferred
#              by 'less' rule. Options are 'greater', 'less'. Default: None.
#         **eval_kwargs: Evaluation arguments fed into the evaluate function of
#             the dataset.
#     """

#     rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
#     init_value_map = {'greater': -inf, 'less': inf}
#     greater_keys = ['mAP', 'AR']
#     less_keys = ['loss']

#     def __init__(self,
#                  dataloader,
#                  start=None,
#                  interval=1,
#                  by_epoch=True,
#                  save_best=None,
#                  rule=None,
#                  **eval_kwargs):
#         if not isinstance(dataloader, DataLoader):
#             raise TypeError('dataloader must be a pytorch DataLoader, but got'
#                             f' {type(dataloader)}')
#         if not interval > 0:
#             raise ValueError(f'interval must be positive, but got {interval}')
#         if start is not None and start < 0:
#             warnings.warn(
#                 f'The evaluation start epoch {start} is smaller than 0, '
#                 f'use 0 instead', UserWarning)
#             start = 0
#         self.dataloader = dataloader
#         self.interval = interval
#         self.by_epoch = by_epoch
#         self.start = start
#         assert isinstance(save_best, str) or save_best is None
#         self.save_best = save_best
#         self.eval_kwargs = eval_kwargs
#         self.initial_epoch_flag = True

#         self.logger = get_root_logger()

#         if self.save_best is not None:
#             self._init_rule(rule, self.save_best)

#     def _init_rule(self, rule, key_indicator):
#         """Initialize rule, key_indicator, comparison_func, and best score.

#         Args:
#             rule (str | None): Comparison rule for best score.
#             key_indicator (str | None): Key indicator to determine the
#                 comparison rule.
#         """
#         if rule not in self.rule_map and rule is not None:
#             raise KeyError(f'rule must be greater, less or None, '
#                            f'but got {rule}.')

#         if rule is None:
#             if key_indicator != 'auto':
#                 if any(key in key_indicator for key in self.greater_keys):
#                     rule = 'greater'
#                 elif any(key in key_indicator for key in self.less_keys):
#                     rule = 'less'
#                 else:
#                     raise ValueError(f'Cannot infer the rule for key '
#                                      f'{key_indicator}, thus a specific rule '
#                                      f'must be specified.')
#         self.rule = rule
#         self.key_indicator = key_indicator
#         if self.rule is not None:
#             self.compare_func = self.rule_map[self.rule]

#     def before_run(self, runner):
#         if self.save_best is not None:
#             if runner.meta is None:
#                 warnings.warn('runner.meta is None. Creating a empty one.')
#                 runner.meta = dict()
#             runner.meta.setdefault('hook_msgs', dict())

#     def before_train_epoch(self, runner):
#         """Evaluate the model only at the start of training."""
#         if not self.initial_epoch_flag:
#             return
#         if self.start is not None and runner.epoch >= self.start:
#             self.after_train_epoch(runner)
#         self.initial_epoch_flag = False

#     def evaluation_flag(self, runner):
#         """Judge whether to perform_evaluation after this epoch.

#         Returns:
#             bool: The flag indicating whether to perform evaluation.
#         """
#         if self.start is None:
#             if not self.every_n_epochs(runner, self.interval):
#                 # No evaluation during the interval epochs.
#                 return False
#         elif (runner.epoch + 1) < self.start:
#             # No evaluation if start is larger than the current epoch.
#             return False
#         else:
#             # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
#             if (runner.epoch + 1 - self.start) % self.interval:
#                 return False
#         return True

#     def after_train_epoch(self, runner):
#         if not self.by_epoch or not self.evaluation_flag(runner):
#             return
#         from mmdet.apis import single_gpu_test
#         results = single_gpu_test(runner.model, self.dataloader, show=False)
#         key_score = self.evaluate(runner, results)
#         if self.save_best:
#             self.save_best_checkpoint(runner, key_score)

#     def after_train_iter(self, runner):
#         if self.by_epoch or not self.every_n_iters(runner, self.interval):
#             return
#         from mmdet.apis import single_gpu_test
#         results = single_gpu_test(runner.model, self.dataloader, show=False)
#         key_score = self.evaluate(runner, results)
#         if self.save_best:
#             self.save_best_checkpoint(runner, key_score)

#     def save_best_checkpoint(self, runner, key_score):
#         best_score = runner.meta['hook_msgs'].get(
#             'best_score', self.init_value_map[self.rule])
#         if self.compare_func(key_score, best_score):
#             best_score = key_score
#             runner.meta['hook_msgs']['best_score'] = best_score
#             # last_ckpt = runner.meta['hook_msgs']['last_ckpt']
#             runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
#             mmcv.symlink(
#                 last_ckpt,
#                 osp.join(runner.work_dir, f'best_{self.key_indicator}.pth'))
#             time_stamp = runner.epoch + 1 if self.by_epoch else runner.iter + 1
#             self.logger.info(f'Now best checkpoint is epoch_{time_stamp}.pth.'
#                              f'Best {self.key_indicator} is {best_score:0.4f}')

#     def evaluate(self, runner, results):
#         eval_res = self.dataloader.dataset.evaluate(
#             results, logger=runner.logger, **self.eval_kwargs)
#         for name, val in eval_res.items():
#             runner.log_buffer.output[name] = val
#         runner.log_buffer.ready = True
#         if self.save_best is not None:
#             if self.key_indicator == 'auto':
#                 # infer from eval_results
#                 self._init_rule(self.rule, list(eval_res.keys())[0])
#             return eval_res[self.key_indicator]
#         else:
#             return None


# class DistEvalHook(EvalHook):
#     """Distributed evaluation hook.

#     Notes:
#         If new arguments are added, tools/test.py may be effected.

#     Attributes:
#         dataloader (DataLoader): A PyTorch dataloader.
#         start (int, optional): Evaluation starting epoch. It enables evaluation
#             before the training starts if ``start`` <= the resuming epoch.
#             If None, whether to evaluate is merely decided by ``interval``.
#             Default: None.
#         interval (int): Evaluation interval (by epochs). Default: 1.
#         tmpdir (str | None): Temporary directory to save the results of all
#             processes. Default: None.
#         gpu_collect (bool): Whether to use gpu or cpu to collect results.
#             Default: False.
#         save_best (str, optional): If a metric is specified, it would measure
#             the best checkpoint during evaluation. The information about best
#             checkpoint would be save in best.json.
#             Options are the evaluation metrics to the test dataset. e.g.,
#             ``bbox_mAP``, ``segm_mAP`` for bbox detection and instance
#             segmentation. ``AR@100`` for proposal recall. If ``save_best`` is
#             ``auto``, the first key will be used. The interval of
#             ``CheckpointHook`` should device EvalHook. Default: None.
#         rule (str | None): Comparison rule for best score. If set to None,
#             it will infer a reasonable rule. Default: 'None'.
#         broadcast_bn_buffer (bool): Whether to broadcast the
#             buffer(running_mean and running_var) of rank 0 to other rank
#             before evaluation. Default: True.
#         **eval_kwargs: Evaluation arguments fed into the evaluate function of
#             the dataset.
#     """

#     def __init__(self,
#                  dataloader,
#                  start=None,
#                  interval=1,
#                  by_epoch=True,
#                  tmpdir=None,
#                  gpu_collect=False,
#                  save_best=None,
#                  rule=None,
#                  broadcast_bn_buffer=True,
#                  **eval_kwargs):
#         super().__init__(
#             dataloader,
#             start=start,
#             interval=interval,
#             by_epoch=by_epoch,
#             save_best=save_best,
#             rule=rule,
#             **eval_kwargs)
#         self.broadcast_bn_buffer = broadcast_bn_buffer
#         self.tmpdir = tmpdir
#         self.gpu_collect = gpu_collect

#     def _broadcast_bn_buffer(self, runner):
#         # Synchronization of BatchNorm's buffer (running_mean
#         # and running_var) is not supported in the DDP of pytorch,
#         # which may cause the inconsistent performance of models in
#         # different ranks, so we broadcast BatchNorm's buffers
#         # of rank 0 to other ranks to avoid this.
#         if self.broadcast_bn_buffer:
#             model = runner.model
#             for name, module in model.named_modules():
#                 if isinstance(module,
#                               _BatchNorm) and module.track_running_stats:
#                     dist.broadcast(module.running_var, 0)
#                     dist.broadcast(module.running_mean, 0)

#     def after_train_epoch(self, runner):
#         if not self.by_epoch or not self.evaluation_flag(runner):
#             return

#         if self.broadcast_bn_buffer:
#             self._broadcast_bn_buffer(runner)

#         from mmdet.apis import multi_gpu_test
#         tmpdir = self.tmpdir
#         if tmpdir is None:
#             tmpdir = osp.join(runner.work_dir, '.eval_hook')
#         results = multi_gpu_test(
#             runner.model,
#             self.dataloader,
#             tmpdir=tmpdir,
#             gpu_collect=self.gpu_collect)
#         if runner.rank == 0:
#             print('\n')
#             key_score = self.evaluate(runner, results)
#             if self.save_best:
#                 self.save_best_checkpoint(runner, key_score)

#     def after_train_iter(self, runner):
#         if self.by_epoch or not self.every_n_iters(runner, self.interval):
#             return

#         if self.broadcast_bn_buffer:
#             self._broadcast_bn_buffer(runner)

#         from mmdet.apis import multi_gpu_test
#         tmpdir = self.tmpdir
#         if tmpdir is None:
#             tmpdir = osp.join(runner.work_dir, '.eval_hook')
#         results = multi_gpu_test(
#             runner.model,
#             self.dataloader,
#             tmpdir=tmpdir,
#             gpu_collect=self.gpu_collect)
#         if runner.rank == 0:
#             print('\n')
#             key_score = self.evaluate(runner, results)
#             if self.save_best:
#                 self.save_best_checkpoint(runner, key_score)

# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class EvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(EvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmdet.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # if self.save_best:
        if self.save_best and key_score:#进行修改了
            self._save_ckpt(runner, key_score)


# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
class DistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
