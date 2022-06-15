import sys

from cv2 import threshold
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, i):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    print(img.shape, flo.shape)

    ############################################################################
    # hsv = np.zeros_like(img)
    # print(hsv.shape)
    # hsv[..., 1] = 255
    # mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # flo_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # flo_gray = cv2.normalize(flo_gray, None, 0, 255, cv2.NORM_MINMAX)
    # th = np.percentile(flo_gray, 2.5) # threshold
    # # print(flo_gray)
    # cv2.imwrite("test.png",flo_gray)
    # print(flo_gray)
    # print(th)
    # w, h = flo_gray.shape
    # Mask = np.zeros((w, h))
    # for x in range(w):
    #     for y in range(h):
    #         if flo_gray[x,y] > th:
    #             Mask[x,y] = 0
    #         else:
    #             Mask[x,y] = 1

    # plt.imshow(Mask, cmap='gray')
    # plt.show()

    ##############################################################################


    # map flow to rgb image
    flo_rgb = flow_viz.flow_to_image(flo)
    flo_gray = cv2.cvtColor(flo_rgb, cv2.COLOR_RGB2GRAY)

    # mask
    w, h = flo_gray.shape
    
    th = np.percentile(flo_gray, 2) # threshold
    print('mean=', th)
    Mask = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            if flo_gray[x,y] > th:
                Mask[x,y] = 0
            else:
                Mask[x,y] = 255
    cv2.imwrite("./demo-frames/raft_flo_rgb/mask_{}.png".format(str(i).zfill(6)), flo_rgb)
    # cv2.imwrite("mask_2.png", Mask)

    # plt.imshow(Mask, cmap='gray')
    # plt.show()



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    i = 1
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, i)
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    demo(args)
