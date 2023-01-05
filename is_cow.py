import sys
from traceback import print_tb
import os
import glob
# sys.path.append('cow_project')
sys.path.append('/workspace')
sys.path.append('/workspace/dusik/cow_detect')
sys.path.append('/workspace/dusik/cow_detect/yolov5')
sys.path.append('/workspace/dusik/cow_detect/yolov5/models')
# sys.path.append('yolov5')
print(sys.path)
from yolov5 import *


# from yolov5.models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.plots import Annotator, colors, save_one_box
# from utils.torch_utils import select_device, smart_inference_mode

from yolov5 import cow_detect

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

def find_cow(folder_check):
    if folder_check == 0:
        check =  os.listdir('/workspace/testFastAPI/test1')
        print(f'=====================================asjdlfkajsdflkajskdlfaskldf')
        if len(check) < 1:
            print('image is not exist')
            return 0
        opt = cow_detect.parse_opt1()
        cow_detect.main(opt)
        flag = cow_detect.temp
        print(flag)
        if False in flag or len(check) != len(flag):
            print('=============================================================not a cow')
            [os.remove(f) for f in glob.glob("/workspace/testFastAPI/test1/*")]
            cow_detect.temp = []
            return False
        else:
            # print(cow_detect.temp)
            cow_detect.temp = []
            return True
    elif folder_check == 1:
        check =  os.listdir('/workspace/testFastAPI/test2')
        print(f'====================================={len(check)}')
        if len(check) < 5:
            print('image is not exist')
            return 0
        opt = cow_detect.parse_opt2()
        cow_detect.main(opt)
        flag = cow_detect.temp
        print(flag)
        if False in flag or len(check) != len(flag):
            print('=============================================================not a cow')
            [os.remove(f) for f in glob.glob("/workspace/testFastAPI/test2/*")]
            cow_detect.temp = []
            return False
        else:
            # print(cow_detect.temp)
            cow_detect.temp = []
            return True