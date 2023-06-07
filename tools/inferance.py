from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import argparse
import glob
import os
from pathlib import Path
import random
import cv2
import shutil
import re
'''


 python tools/inferance.py configs/swin/swin_t_yolox.py  work_dirs/swin_t_yolox/best_bbox_mAP_epoch_8332.pth --data_path test/ele/ --device cuda --local_rank 0 --output test/swintyolo/ele
 python tools/inferance.py configs/swin/swin_t_yolox.py  work_dirs/swin_t_yolox/best_bbox_mAP_epoch_8332.pth --data_path test/exam/  --output test/swintyolo/exam
 
 
 '''
def parse_args():
        parser = argparse.ArgumentParser(description='inferance')
        parser.add_argument('config',default='configs/swin/swin_l_yolox.py', help='test config file path')
        parser.add_argument('checkpoint',default='work_dirs/swin_l_yolox/best_bbox_mAP_epoch_5642.pth', help='checkpoint file')
        parser.add_argument('--data_path',type=str,default='test/img',help=' data path')
        parser.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='device')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--output',type=str,default='test/swinlyolox',help='output folder')
        parser.add_argument('--save-txt',action='store_true',help='save result to txt')
        args=parser.parse_args()
        return args
    
def get_data(dir):
    if Path(dir):
        data=sorted(glob.glob(os.path.join(dir,'*.jpg')),key=lambda x:int(re.match('\D*(\d+)',x).group(1)))
    return data

def main():
    args=parse_args()
    imgs=get_data(args.data_path)
    model=init_detector(args.config,args.checkpoint,args.device)
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    classes=['normal','overlap']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]#每个类别对应的颜色框
    cluster_num=0
    for img in imgs:
        tmp=0
        result=inference_detector(model,img)
        #list[array([[左上横坐标，左上纵坐标，右下横坐标，右下纵坐标，置信度]])] 
        '''
        result[0]为正常的团簇的结果
        result[1]为重叠的团簇结果
        '''
        
        if args.save_txt:
            txt_path=str(Path(args.output)/Path('res.txt'))
            with open(txt_path,'w') as f:
                f.write(result)
                
        save_path=str(Path(args.output)/Path(img).name)
        src=cv2.imread(img)
        for cls in range(len(result)):
            for x1,y1,x2,y2,conf in result[cls]:
                if conf>=0.25:
                    label = '%s %.2f' % (classes[cls], conf)
                    cluster_num+=1
                    tmp+=1
                    plot_one_box((x1,y1,x2,y2),src,color=colors[cls],label=label,line_thickness=2)
                
        cv2.imwrite(save_path,src)
        print('file:{};cluster:{}'.format(img,tmp))      
        # show_result_pyplot(model,img,result)
    print('cluster number:{}'.format(cluster_num))    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=0.4, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, 1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.4, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        
if __name__ == '__main__':
    main()