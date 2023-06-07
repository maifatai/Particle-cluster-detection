import argparse
import numpy as np
import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import time
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,replace_ImageToTensor)

from ptflops import get_model_complexity_info
from thop import profile

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 128],
        help='input image size')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--optimal_batch_size',default=1,help='calculate Throughput for optimal batch size')
    parser.add_argument('--img',type=str,default='test/img/1.jpg', help='Image file')
    parser.add_argument('--checkpoint', help='Checkpoint file')#权重文件，用于计算FPS
    parser.add_argument('--device', default='cuda:0', help='Device used for FPS')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--save',type=str,default='res.txt')
    args = parser.parse_args()
    return args

def cal_FPS(args,cfg):
    # args = parse_args()
    # cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    out=None
    # benchmark with 2000 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            out=model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ 1000], fps: {fps:.1f} img / s')

        if (i + 1) == 1000:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break
    print('out:',out)
    print('type out:',type(out))
    return fps

def cal_Throughput(args,model):
    optimal_batch_size=args.optimal_batch_size#max_batch_size for GPU
    Throughput=0
    j=0
    while j<200:
        try:
            dummy_input = torch.randn(optimal_batch_size, 3,512,128, dtype=torch.float).cuda()
            repetitions=100
            total_time = 0
            with torch.no_grad():
                for rep in range(repetitions):
                    starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    starter.record()
                    out = model(dummy_input)
                    ender.record()
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)/1000
                    total_time += curr_time
            Throughput = max((repetitions*optimal_batch_size)/total_time, Throughput)
            print("optimal_batch_size",optimal_batch_size,'Throughput:',Throughput)
            optimal_batch_size+=1
            
            j+=1
        except RuntimeError as e:#(out of memory)
            if 'cuda out of memory' in str(e).lower:
                print('Final Throughput:',Throughput)
                return Throughput
            else:
                raise e
    return Throughput

def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    #使用 ptflops：https://github.com/sovrasov/flops-counter.pytorch
    flops, params = get_model_complexity_info(model, input_shape)
    print('Flops:  ' + flops)
    print('Params: ' + params)
   
    input = torch.randn(1, 3, 512, 128).cuda()
    flops1, params1 = profile(model, inputs=(input, ))
    print("FLOPs=", str(flops1/1e9) +'{}'.format("G")) #1GFPLOs = 10^9FLOPs
    print("params=", str(params1/1e6)+'{}'.format("M"))
    
    #计算FPS
    fps=cal_FPS(args,cfg)

    #计算Throughput
    # Throughput=cal_Throughput(args,model)
    Throughput=1
    # optimal_batch_size=150#max_batch_size for GPU
    # Throughput=0
    # while True:
    #     try:
    #         dummy_input = torch.randn(optimal_batch_size, 3,512,128, dtype=torch.float).cuda()
    #         repetitions=100
    #         total_time = 0
    #         with torch.no_grad():
    #             for rep in range(repetitions):
    #                 starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    #                 starter.record()
    #                 out = model(dummy_input)
    #                 ender.record()
    #                 torch.cuda.synchronize()
    #                 curr_time = starter.elapsed_time(ender)/1000
    #                 total_time += curr_time
    #         Throughput = max((repetitions*optimal_batch_size)/total_time, Throughput)
    #         optimal_batch_size+=1
    #         print('Throughput:',Throughput)
    #     except:#(out of memory)
    #         break
    # print('Final Throughput:',Throughput)
    
    #计算FPS
    # model1=init_detector(args.config, args.checkpoint, device=args.device)
    # total_time = 0
    # for i in range(100):
    #     starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    #     starter.record()
    #     # test a single image
    #     result = inference_detector(model1, args.img)
    #     # show the results
    #     show_result_pyplot(model1, args.img, result, score_thr=args.score_thr)
    #     ender.record()
    #     torch.cuda.synchronize()
    #     curr_time = starter.elapsed_time(ender)/1000
    #     total_time += curr_time
    # FPS=1.0/(total_time/100)
    # print("FPS: %f"%(1.0/(total_time/100)))
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print("FLOPs=", str(flops1/1e9) +'{}'.format("G")) #1GFPLOs = 10^9FLOPs
    print("params=", str(params1/1e6)+'{}'.format("M"))
    print(f'Overall fps: {fps:.1f} img / s')
    print('Final Throughput:',Throughput)
    path=args.checkpoint.replace('latest.pth',args.save)
    with open(path,'w') as f:
        data=str({'Flops:  ':flops,'Params: ':params,"FLOPs":flops1,"params":params1,'fps':fps,'Throughput':Throughput})
        f.write(data)
    
if __name__ == '__main__':
    main()
