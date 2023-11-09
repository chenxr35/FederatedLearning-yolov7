import os
import numpy as np
import torch
from utils.callbacks import EvalCallback
from utils.utils import get_anchors, get_classes
from nets.yolo import YoloBody
from nets.yolo_training import weights_init

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    eval_period = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0

    Cuda = True

    pretrained = True

    input_shape = [640, 640]

    phi = 'l'

    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    val_annotation_path = '2007_test.txt'

    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    model_path = 'logs/fedavg_5_iidFalse/checkpoints_2023_01_30_07_40_22/best_epoch_weights.pth'

    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, phi, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    model.cuda()

    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                             temp_dir, Cuda, map_out_path=temp_dir, eval_flag=True, period=eval_period)

    model_eval = model.eval()

    eval_callback.on_epoch_end(eval_period, model_eval)