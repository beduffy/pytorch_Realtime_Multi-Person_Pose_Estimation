# Adapted from https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation

import os
import re
import sys
import torch, cv2
import math
import time
import scipy
import argparse
import matplotlib
from torch import np
import pylab as plt
from joblib import Parallel, delayed
import util
# import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

# parser = argparse.ArgumentParser()
# parser.add_argument('--t7_file', required=True)
# parser.add_argument('--pth_file', required=True)
# args = parser.parse_args()

# from gtts import gTTS
# import pyttsx
# engine = pyttsx.init()
# engine.say('Good morning.')
# engine.runAndWait()

torch.set_num_threads(torch.get_num_threads())

import main_functions



# hands_over_head = True
# how_many_times_hands_went_over_head = 0
#global how_many_times_hands_went_over_head




if __name__ == "__main__":
    print('Setting up model')
    model, model_params = main_functions.setup_model()

    jjac_info = {
        'num_jumping_jacks': 0,
        'hands_over_head': False,
        'biggest_diff': 0,
        'last_x_head_pos': []
    }
    _ = main_functions.process_image(np.ones((320, 320, 3)), model, model_params, jjac_info)

    # webcam = True
    webcam = False
    # for webcam

    if webcam:
        cap = cv2.VideoCapture(0)   ##############
    # for video
    else:
        cap = cv2.VideoCapture('data/Jumping_Jacks.mp4')

    pause = False
    called_message = False
    key_wait_time = 1
    frame_jump_interval = 1 if webcam else 4

    counter = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        if not pause:
            #if called_message:
            if not webcam:
                cap.set(1, counter)
            ret, frame = cap.read()

            counter += frame_jump_interval # 5x faster but 4x is maybe safer
            if not webcam:
                if counter < 200:
                    continue

            if counter % 3 == 0:
                canvas = main_functions.process_image(frame,  model, model_params, jjac_info)
            else:
                canvas = frame
        else:
            if not called_message:
                called_message = True
                print('pause is True')

            pass
            # if not canvas.any():
            #     canvas = frame


        # Display the resulting frame
        #canvas = frame
        cv2.imshow('Video', canvas)

        #time.sleep(1)
        k = cv2.waitKey(key_wait_time)
        if k & 0xFF == ord('q'):
            break

        if k & 0xFF == ord('p'):
            pause = True
            #pause = True

        if k & 0xFF == ord('o'):
            pause = False

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
