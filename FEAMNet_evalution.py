# -*- coding: utf-8 -*-


import numpy as np
import os
import time
from FEAMNet_func.util import write_pfm, read_pfm
from FEAMNet_func.util import make_epiinput
from FEAMNet_func.util import make_input
from FEAMNet_func.func_model_81 import define_FEAMNet

import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':



    dir_output = 'FEAMNet_output'

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


    LFdir = 'synthetic'

    if (LFdir == 'synthetic'):
        dir_LFimages = ['hci_dataset/stratified/backgammon', 'hci_dataset/stratified/dots',
                        'hci_dataset/stratified/pyramids', 'hci_dataset/stratified/stripes',
                        'hci_dataset/training/boxes', 'hci_dataset/training/cotton',
                        'hci_dataset/training/dino','hci_dataset/training/sideboard']

        image_w = 512
        image_h = 512


    AngualrViews = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # number of views ( 0~8 for 9x9 )

    
    path_weight='./FEAMNet_checkpoint/FEAMNet_ckp/iter0026_valmse1.087_bp2.95.hdf5'

    img_scale = 1  

    img_scale_inv = int(1 / img_scale)


    model_learning_rate = 0.001
    model_512 = define_FEAMNet(round(img_scale * image_h),
                              round(img_scale * image_w),
                              AngualrViews,
                              model_learning_rate)



    model_512.load_weights(path_weight)
    dum_sz = model_512.input_shape[0]
    dum = np.zeros((1, dum_sz[1], dum_sz[2], dum_sz[3]), dtype=np.float32)
    tmp_list = []
    for i in range(81):
        tmp_list.append(dum)
    dummy = model_512.predict(tmp_list, batch_size=1)

    avg_attention = []

    """  Depth Estimation  """
    for image_path in dir_LFimages:
        val_list = make_input(image_path, image_h, image_w, AngualrViews)

        start = time.clock()

        val_output_tmp, attention_tmp = model_512.predict(val_list, batch_size=1)

        runtime = time.clock() - start

        print("runtime: %.5f(s)" % runtime)


        avg_attention.append(np.reshape(attention_tmp[0, 0, 0, 0], (9, 9)))
        cv2.imwrite('attention_'+image_path.split('/')[-1]+'.png', cv2.resize(np.reshape(attention_tmp[0, 0, 0, 0], (9, 9)) * 255.0, (0, 0), fx=25, fy=25, interpolation=cv2.INTER_NEAREST).astype(np.uint8))


        write_pfm(val_output_tmp[0, :, :], dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        print('pfm file saved in %s/%s.pfm' % (dir_output, image_path.split('/')[-1]))


    avg_attention_map = np.stack(avg_attention, 0)
    avg_attention_map = np.mean(avg_attention_map, 0)
    
    cv2.imwrite('attention.png', cv2.resize(avg_attention_map * 255.0, (0, 0), fx=25, fy=25, interpolation=cv2.INTER_NEAREST).astype(np.uint8))


    output_stack = []
    gt_stack = []
    for image_path in dir_LFimages:
        output = read_pfm(dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
        gt = read_pfm(image_path + '/gt_disp_lowres.pfm')
        gt_490 = gt[15:-15, 15:-15]
        output_stack.append(output)
        gt_stack.append(gt_490)
    output = np.stack(output_stack, 0)
    gt = np.stack(gt_stack, 0)

    output = output[:, 15:-15, 15:-15]

    train_diff = np.abs(output - gt)
    train_bp = (train_diff >= 0.07)

    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    training_bad_pixel_ratio = 100 * np.average(train_bp)

    print('Pre-trained Model average MSE*100 = %f' % training_mean_squared_error_x100)
    print('Pre-trained Model average Badpix0.07 = %f' % training_bad_pixel_ratio)


for image_path in dir_LFimages:
    output = read_pfm(dir_output + '/%s.pfm' % (image_path.split('/')[-1]))
    gt = read_pfm(image_path + '/gt_disp_lowres.pfm')
    
    gt_490 = gt[15:-15, 15:-15]
    output_490 = output[15:-15, 15:-15]
    
    train_diff = np.abs(output_490 - gt_490)
    train_bp = (train_diff >= 0.07)
    train_bp3 = (train_diff >= 0.03)
    train_bp1 = (train_diff >= 0.01)

    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    training_bad_pixel_ratio = 100 * np.average(train_bp)
    #training_bad_pixel_ratio3 = 100 * np.average(train_bp3)
    #training_bad_pixel_ratio1 = 100 * np.average(train_bp1)

    print("*" * 10 + image_path.split('/')[-1] + "*" * 10)
    print('Pre-trained Model average MSE*100 = %f' % training_mean_squared_error_x100)
    print('Pre-trained Model average Badpix0.07 = %f' % training_bad_pixel_ratio)
    #print('Pre-trained Model average Badpix0.03 = %f' % training_bad_pixel_ratio3)
    #print('Pre-trained Model average Badpix0.01 = %f' % training_bad_pixel_ratio1)
