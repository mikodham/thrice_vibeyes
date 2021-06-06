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

# Dataset, Models
dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)
# Parser
parser = argparse.ArgumentParser(description='Testing pan generation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data', metavar='DIR', default=os.getcwd(), help='path to dataset')
parser.add_argument('-tn', '--tdataName', metavar='Test Data Set Name', default='Kitti_eigen_test_improved',
                    choices=dataset_names)
parser.add_argument('-relbase', '--rel_baselne', default=1, help='Relative baseline of testing dataset')
parser.add_argument('-mdisp', '--max_disp', default=300)  # of the training patch W
parser.add_argument('-mindisp', '--min_disp', default=2)  # of the training patch W
parser.add_argument('-b', '--batch_size', metavar='Batch Size', default=1)
parser.add_argument('-eval', '--evaluate', default=True)
parser.add_argument('-save', '--save', default=False)
parser.add_argument('-save_pc', '--save_pc', default=False)
parser.add_argument('-save_pan', '--save_pan', default=False)
parser.add_argument('-save_input', '--save_input', default=False)
parser.add_argument('-w', '--workers', metavar='Workers', default=4)
parser.add_argument('--sparse', default=False, action='store_true',
                    help='Depth GT is sparse, automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('-gpu_no', '--gpu_no', default='1', help='Select your GPU ID, if you have multiple GPU.')
parser.add_argument('-dt', '--dataset', help='Dataset and training stage directory', default='Kitti_stage2')
parser.add_argument('-ts', '--time_stamp', help='Model timestamp', default='03-29-14_18')
parser.add_argument('-m', '--model', help='Model', default='FAL_netB')
parser.add_argument('-no_levels', '--no_levels', default=49, help='Number of quantization levels in MED')
parser.add_argument('-dtl', '--details', help='details',
                    default=',e20es,b4,lr5e-05/checkpoint.pth.tar')
parser.add_argument('-fpp', '--f_post_process', default=False, help='Post-processing with flipped input')
parser.add_argument('-mspp', '--ms_post_process', default=True, help='Post-processing with multi-scale input')
parser.add_argument('-median', '--median', default=False,
                    help='use median scaling (not needed when training from stereo')

input_transform = transforms.Compose([
    # data_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),  # (input - mean) / std
    transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
])

target_transform = transforms.Compose([
    # data_transforms.ArrayToTensor(),
    transforms.Normalize(mean=[0], std=[1]),
])


def display_config(save_path):
    settings = ''
    settings = settings + '############################################################\n'
    settings = settings + '# FAL-net        -         Pytorch implementation          #\n'
    settings = settings + '# by Juan Luis Gonzalez   juanluisgb@kaist.ac.kr           #\n'
    settings = settings + '############################################################\n'
    settings = settings + '-------YOUR TRAINING SETTINGS---------\n'
    for arg in vars(args):
        settings = settings + "%15s: %s\n" % (str(arg), str(getattr(args, arg)))
    print(settings)
    # Save config in txt file
    with open(os.path.join(save_path, 'settings.txt'), 'w+') as f:
        f.write(settings)


def ms_pp(input_view, pan_model, flip_grid, disp, min_disp, max_pix):
    B, C, H, W = input_view.shape

    up_fac = 2/3
    upscaled = F.interpolate(F.grid_sample(input_view, flip_grid), scale_factor=up_fac, mode='bilinear',
                             align_corners=True)
    dwn_flip_disp = pan_model(upscaled, min_disp, max_pix, ret_disp=True, ret_pan=False, ret_subocc=False)
    dwn_flip_disp = (1 / up_fac) * F.interpolate(dwn_flip_disp, size=(H, W), mode='nearest')  # , align_corners=True)
    dwn_flip_disp = F.grid_sample(dwn_flip_disp, flip_grid)

    norm = disp / (np.percentile(disp.detach().cpu().numpy(), 95) + 1e-6)
    norm[norm > 1] = 1

    return (1 - norm) * disp + norm * dwn_flip_disp


def frametodisp(frame):
    # print("-------------GET DISPARITY--------------")
    input_left = frame  # INITIAL: (720, 1280, 3)
    # print("TYPE ", type(input_left))
    # right_shift = args.max_disp * args.rel_baselne
    # Convert min and max disp to bx1x1 tensors
    # global variable
    max_disp = torch.tensor([1])  # torch.Tensor([right_shift]).unsqueeze(1).unsqueeze(1).type(input_left.type())
    min_disp = torch.tensor([255])  # max_disp * args.min_disp / args.max_disp


    input_left = torch.moveaxis(input_left, 2, 0)
    input_left = input_transform(input_left)
    input_left = torch.unsqueeze(input_left, 0)
    input_left = torch.tensor(input_left).cuda()
    # Prepare flip grid for post-processing
    B, C, H, W = input_left.shape
    i_tetha = torch.zeros(B, 2, 3).cuda()
    i_tetha[:, 0, 0] = 1
    i_tetha[:, 1, 1] = 1
    flip_grid = F.affine_grid(i_tetha, [B, C, H, W])
    flip_grid[:, :, :, 0] = -flip_grid[:, :, :, 0]

    print("INPUT LEFT TYPE ", type(input_left), input_left.shape)
    # frame = np.expand_dims(frame, 0)    # SHAPE:  (1, 3, 720, 1280)
    # to satisfy B, C, H, W = input_left.shape
    disp = pan_model(input_left, min_disp, max_disp, ret_disp=True, ret_subocc=False, ret_pan=False)
    # print("DISP IS ", disp.shape)
    # print(disp)

    # Multi scale input post processing
    disp = ms_pp(input_left, pan_model, flip_grid, disp, min_disp, max_disp)

    return disp


def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    # if hasFrames:
    #     cv2.imwrite("image" + str(count) + ".jpg", image)  # save frame as JPG file
    # print("TYPE OF IMAGE", type(image))
    return hasFrames, image


if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_/no

    print('--------Testing on gpu ' + args.gpu_no + '-------')
    save_path = os.path.join('Test_Results', args.tdataName, args.model, args.time_stamp)
    if args.f_post_process:
        save_path = save_path + 'fpp'
    if args.ms_post_process:
        save_path = save_path + 'mspp'
    print('=> Saving to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    display_config(save_path)

    print("-------------PAN_MODEL--------------")
    # create pan model
    model_dir = os.path.join(args.dataset, args.time_stamp, args.model + args.details)
    model_dir = os.path.join(os.getcwd(), "pretrained", model_dir)
    print("model_dir: ", model_dir)
    pan_network_data = torch.load(model_dir)  # map_location={'cuda:0': 'cpu'}
    pan_model = pan_network_data['m_model']
    print("=> using pre-trained model for pan '{}'".format(pan_model))
    pan_model = models.__dict__[pan_model](pan_network_data, no_levels=args.no_levels).cuda()
    pan_model = torch.nn.DataParallel(pan_model, device_ids=[0]).cuda()

    print("-------------DEPTH ESTIMATE--------------")
    vid_file = 'S10P_IdeaFactory1'  # 'S10P_NoZoom.mp4'  #
    vid_directory = os.path.join(os.getcwd(), 'Video_Testing', vid_file + '.mp4')
    vidcap = cv2.VideoCapture(vid_directory)
    sec = 0
    frameRate = 0.5  # //it will capture image in each 0.5 second
    count = 1
    results = []
    success, frame = getFrame(sec)
    print("INITIAL:", frame.shape)  # INITIAL: (720, 1280, 3)
    with torch.no_grad():
        while success:
            print("FRAME STARTS COUNT ", count)
            sec = sec + frameRate
            sec = round(sec, 2)
            success, frame = getFrame(sec)
            if success is False:
                break
            # print("SHAPE: ", frame.shape)       # INITIAL SHAPE:  (720, 1280, 3)
            # frame = np.swapaxes(frame,0,2)      # SHAPE:  (3, 1280, 720)
            # frame = np.swapaxes(frame, 1, 2)    # SHAPE:  (3, 720, 1280)

            # TODO: Reduce frame resolution
            frame = cv2.resize(frame, (480, 140), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            torch_frame = torch.from_numpy(frame)
            torch_frame = torch_frame.type(torch.FloatTensor)
            prev = time.time()
            disp = frametodisp(torch_frame)  # return <class 'torch.Tensor'>, torch.Size([1, 1, 720, 1280])
            disparity = disp.squeeze().cpu().numpy()
            disparity = 256 * np.clip(disparity / (np.percentile(disparity, 95) + 1e-6), 0, 1)
            print("TIME SPENT ", time.time()-prev)
            results.append(disparity)
            count = count + 1

    print("--------DISPARITY->RGB-----------")
    dataset_gif = []
    for each_disp in results:
        # frame = np.squeeze(each_disp.detach().cpu().numpy())
        frame = np.array(each_disp, dtype=np.uint8)
        # print("INITIAL DISPARITY:", frame.shape)
        img = frame
        heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # heatmap = frame
        # print("RESULT SHAPE", heatmap.shape)
        result = np.moveaxis(heatmap, 2, 0)
        dataset_gif.append(result)
    print(dataset_gif[0].shape)
    print(len(dataset_gif))

    print("--------SAVING_GIF-----------")
    from array2gif import write_gif
    gif_name = '06-02' + vid_file + str(time.time()) + '.gif'
    gif_dir = os.path.join(os.getcwd(), "Video_Testing", "gif", gif_name)
    write_gif(dataset_gif, gif_dir, fps=1/frameRate)
    #: param dataset: A NumPy array or list of arrays with shape rgb x rows x cols and integer values in [0, 255].
    print('---------------FINISH------------')
    '''
        Results is a List of Disparity, where disparity is torch.tensor datatype
        Dataset_gif is the list of HeatMaps, which is np.array datatype
    '''
