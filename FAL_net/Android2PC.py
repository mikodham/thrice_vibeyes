import urllib3
import urllib.request
import cv2
import numpy as np
import ssl
import time
import Miko_Testing
import argparse
import time
import numpy as np
from imageio import imsave
import matplotlib.pyplot as plt
from PIL import Image
import os
import Datasets
import models
import torch
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import myUtils as utils
import data_transforms
# from loss_functions import realEPE
import cv2
import time
import depthlevel.dep2level as dl2

# BOOTING
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

parser = Miko_Testing.init()
args = parser.parse_args()
Miko_Testing.test_GPU(args)
pan_model = Miko_Testing.load_model(args)

url = 'http://143.248.252.57:8080/video'
cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.startWindowThread()
cv2.namedWindow("DisparityMap")
cv2.startWindowThread()
cv2.namedWindow("LevelMap")
# https://stackoverflow.com/questions/53130370/how-to-access-phone-camera-using-python-script/53131060
results=[]

c=0
SIZE = (480,260)

# MAIN
while True:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1);
    ret, frame = cap.read()
    prev = time.time()
    if frame is not None:
        cv2.imshow("preview", cv2.resize(frame,SIZE))

        # -------------------------------------
        frame = cv2.resize(frame, SIZE, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        torch_frame = torch.from_numpy(frame)
        torch_frame = torch_frame.type(torch.FloatTensor)

        disp = Miko_Testing.frametodisp(pan_model, torch_frame)  # return <class 'torch.Tensor'>, torch.Size([1, 1, 720, 1280])
        # disparity = disp.squeeze().cpu().numpy()
        disparity = disp.squeeze().cpu().detach().numpy()
        disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
        '''
        # FOR TESTING THE RANGE
        print(disparity.shape)
        for ka in range(0,260, 10):
            # many = np.count_nonzero(ka<= disparity <ka+10, axis = 0)
            many = 0
            for ax in range(400):
                many += (np.logical_and(ka<=disparity[:, ax], disparity[:, ax]<ka+10) ).sum()
            print("range is %d - %d: %d" % (ka, ka+10, many))
        break
        '''
        level = 255/9 * dl2.dep2level(disparity)
        # -------------------------------------
        print("Count", c, "TIME SPENT ", time.time() - prev)
        results.append(disparity)
        cv2.imshow("DisparityMap", cv2.applyColorMap(np.array(disparity, dtype=np.uint8), cv2.COLORMAP_JET))
        q = cv2.waitKey(1)
        # cv2.resizeWindow('LevelMap', SIZE)
        cv2.imshow("LevelMap", cv2.applyColorMap(np.array(level, dtype=np.uint8), cv2.COLORMAP_JET))
    if q == ord("q"):
        break
    c += 1
cv2.destroyAllWindows()
