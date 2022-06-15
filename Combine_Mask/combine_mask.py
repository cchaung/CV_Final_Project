"""
將Mask_RCNN以及raft的mask進行疊加，生成出最終mask
"""

import os
import cv2 
from matplotlib import pyplot as plt
from PIL import Image

if __name__ == "__main__":
    
    for i in range(1,46):
        """
        RCNN_mask_path: Mask_RCNN的結果存放路徑
        raft_path: raft的結果存放路徑
        """

        RCNN_mask_path = r".\video_clip_mask\0000{}.png".format(str(i).zfill(2))
        raft_path = r".\raft\mask_0000{}.png".format(str(i).zfill(2))
        
        mask1 = cv2.imread(RCNN_mask_path, 0)
        
        _, mask1  = cv2.threshold(mask1 , 1, 255, cv2.THRESH_BINARY)
        
        # plt.imshow(mask1, cmap="gray")
        # plt.show()
        
        mask2 = cv2.imread(raft_path, 0)
        mask2 = cv2.resize( mask2, tuple(reversed(mask1.shape)))
        _, mask2  = cv2.threshold(mask2 , 1, 255, cv2.THRESH_BINARY)
        
        # plt.imshow(mask2, cmap="gray")    
        # plt.show()
        
        將疊加的結果
        _, mask  = cv2.threshold(mask1+mask2, 1, 255, cv2.THRESH_BINARY)
        
        """
        使用close運算來降低雜訊點
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        plt.imshow(mask, cmap="gray")    
        
        mask = Image.fromarray(mask)
        # mask.show()
        mask.save("final_mask/0000{}.png".format(str(i).zfill(2)))
        # plt.show()