#!/bin/bash



# swin_b
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_b_rcnn/swin_b_rcnn.log.json --out work_dirs/swin_b_rcnn/loss.jpg --keys loss --title loss



# swin_l
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_rcnn/swin_l_rcnn.log.json   --out work_dirs/swin_l_rcnn/loss.jpg --keys loss --title loss



# swin_s
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_s_rcnn/swin_s_rcnn.log.json --out work_dirs/swin_s_rcnn/loss.jpg --keys loss --title loss


# swin_t
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_t_rcnn/swin_t_rcnn.log.json --out work_dirs/swin_t_rcnn/loss.jpg --keys loss --title loss


# swin_b_yolo
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_b_yolox/swin_b_yolox.log.json --out work_dirs/swin_b_yolox/loss.jpg --keys loss --title loss

#python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_l_yolox/swin_t_yolox.log.json --out work_dirs/swin_l_yolox/loss_obj.jpg --keys loss_obj --title loss_obj

# swin_l_yolo
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/swin_l_yolox/swin_l_yolox.log.json --out work_dirs/swin_l_yolox/loss.jpg --keys loss --title loss


# swin_s_yolo
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_s_yolox/swin_s_yolox.log.json --out work_dirs/swin_s_yolox/loss.jpg --keys loss --title loss

# swin_t_yolo
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/swin_t_yolox/swin_t_yolox.log.json --out work_dirs/swin_t_yolox/loss.jpg --keys loss --title loss




# convnext_b_yolo
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_b_yolox/convnext_b_yolox.log.json --out work_dirs/convnext_b_yolox/loss.jpg --keys loss --title loss

#convnext_l_yolo
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_l_yolox/convnext_l_yolox.log.json --out work_dirs/convnext_l_yolox/loss.jpg --keys loss --title loss


#convnext_s_yolo
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_s_yolox/convnext_s_yolox.log.json --out work_dirs/convnext_s_yolox/loss.jpg --keys loss --title loss


# convnext_t_yolo
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/loss_obj.jpg --keys loss_obj --title loss_obj
python tools/analysis_tools/analyze_logs.py plot_curve   work_dirs/convnext_t_yolox/convnext_t_yolox.log.json --out work_dirs/convnext_t_yolox/loss.jpg --keys loss --title loss



# convnext_b
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_b_rcnn/convnext_b_rcnn.log.json --out work_dirs/convnext_b_rcnn/loss.jpg --keys loss --title loss


# convnext_l
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/convnext_l_rcnn.log.json --out work_dirs/convnext_l_rcnn/loss.jpg --keys loss --title loss

#python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_l_rcnn/20221113_014435.log.json --out work_dirs/convnext_l_rcnn/loss.jpg --keys loss --title loss


# convnext_s
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_s_rcnn/convnext_s_rcnn.log.json --out work_dirs/convnext_s_rcnn/loss.jpg --keys loss --title loss

# convnext_t
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/bbox_mAP.jpg --keys bbox_mAP --title bbox_mAP
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/bbox_mAP_50.jpg --keys bbox_mAP_50 --title bbox_mAP_50
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/bbox_mAP_75.jpg --keys bbox_mAP_75 --title bbox_mAP_75
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/loss_rpn_cls.jpg --keys loss_rpn_cls --title loss_rpn_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/loss_rpn_bbox.jpg --keys loss_rpn_bbox --title loss_rpn_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/loss_cls.jpg --keys loss_cls --title loss_cls
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/acc.jpg --keys acc --title acc
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/loss_bbox.jpg --keys loss_bbox --title loss_bbox
python tools/analysis_tools/analyze_logs.py plot_curve  work_dirs/convnext_t_rcnn/convnext_t_rcnn.log.json --out work_dirs/convnext_t_rcnn/loss.jpg --keys loss --title loss


