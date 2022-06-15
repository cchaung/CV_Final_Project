# Object and its effects eliminate

###### tags: `Mask R-CNN` `RAFT` `Deepfillv2`

## Member
* **數據所 310554031 葉詠富**
* **數據所 310554037 黃乾哲**


## Workflow

![](https://i.imgur.com/gcJarFW.png)



## Mask R-CNN
* We use the Mask RCNN to select the object.
* The training data is “PennFudanPed” which is containing images that are used for pedestrian detection in the experiments reported in.

![](https://i.imgur.com/Vw51gv9.png)

* Since our background is sample, the human contour is clear. 
* We do the threshold to strengthen the the curve of the mask

![](https://i.imgur.com/B0tUpZM.png)

 



## RAFT 

### Flow to RGB to Grayscale
* Use the pretrain model raft-thing to detect the flow from people and shadow 
* convert flow to rgb images

![](https://i.imgur.com/FjPjIBn.png)

* convert rgb image  to grayscale image

![](https://i.imgur.com/Z0GMpW2.png)

### RAFT Mask by Threshold
* Mapping grayscale images onto a 1D vector
* Set the threshold to create RAFT Mask.
* The threshold is set with percentiles.
* For example: 10% of the data is less than the threshold as the foreground, greater than the threshold as the background

![](https://i.imgur.com/U8yRQTv.png)

* threshold 2%  vs 10%
    * 2%   covers less shadows, but reduces the effect of optical flow in nearby scenes
    * 10% covers more shadows, but increases the effect of the optical flow of nearby scenes
    * We choose 10%, hoping to cover more shadows

![](https://i.imgur.com/sRboRaE.png)


### RAFT Mask by KMeans
* Divide the rgb optical flow map into 10 groups using Kmeans
* Sort by the number of pixels in the group
* If the difference between the front and the back is 5 times, it is the dividing point between the foreground and background.
* Example: 90505/4247=21 > 5

![](https://i.imgur.com/SenbeJS.png)

### Threshold v.s. KMeans
* We can find that the use of KMeans can better connect the relationship between people and shadows to generate masks.
* Eliminate the effects of background.
* The people behind the original were kept.

![](https://i.imgur.com/uBtLXaZ.png)




## Combine Mask
* Mask RCNN is good at detecting the object.
* RAFT is good at detecting the shadow.
* Combine the both advantage to produce the mask.

![](https://i.imgur.com/vcKR8yl.png)


## Deepfillv2
* We use pretrain model Deepfillv2 to inpainting disappaer part by the mask

![](https://i.imgur.com/5xPjarV.png)



## Discussion
* The mask almost cover the whole object and its effect.
* RAFT can not detect the whole shadow, it may cause by the action of the object.
* Inpainting result is not nature, it need to retrain a personal model for this video’s background.


## Reference
* Mask R-CNN
    * https://www.youtube.com/watch?v=5VLI_gbpocE&t=1s&ab_channel=WilsonHo
    * https://zhuanlan.zhihu.com/p/142757151
    * https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
* RAFT
    * https://github.com/princeton-vl/RAFT
    * https://arxiv.org/pdf/2003.12039.pdf
* Deepfillv2
    * https://github.com/nipponjo/deepfillv2-pytorch
* KMeans
    * https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python
    * https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/
    * https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html



