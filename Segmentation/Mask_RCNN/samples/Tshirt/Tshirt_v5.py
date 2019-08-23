"""
Mask R-CNN
Train on the toy Tshirt dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Modified by Seoyoon Park
------------------------------------------------------------
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/chief/FittingroomAnywhere/segmentation/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

#BalloonConfig->TshirtConfig
class TshirtConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Tshirt"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Tshirt

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.98


############################################################
#  Dataset
############################################################

# BalloonDataset -> TshirtDataset
class TshirtDataset(utils.Dataset):
    # 이미지에 대해 인스턴스 마스크를 생성하는 함수
    # 리턴 :
    #      (1) masks : [이미지 높이, 이미지 너비, 인스턴스 수] shape의 bool array.
    #          인스턴스 하나 당 하나의 마스크
    #      (2) class_ids : 인스턴스 마스크의 클래스 아이디로 이루어진 1D array
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Thsirt dataset image, delegate to parent class.
        # self.image_info 부분은 mrcnn/utils.py의 add_image() 함수를 참고하라.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Tshirt":
            return super(self.__class__, self).load_mask(image_id)


        # polygon을 비트맵 마스크로 바꾸는 부분
        # 마스크 shape는 [height, width, instance_count]
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # mask, 각 인스턴스의 클래스 ID에 대한 array (shape=[instance_count])를 리턴한다.
        # 우리는 오직 하나의 클래스 아이디만 가지고 있기 때문에, 1로 이루어진 어레이를 리턴할 것
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    # 이미지의 path를 리턴한다.
    def image_reference(self, image_id):
        """Return the path of the image."""
        # self.image_info 부분은 mrcnn/utils.py의 add_image() 함수를 참고하라.
        info = self.image_info[image_id]
        if info["source"] == "Tshirt":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

########## End of class TshirtDataset(utils.Dataset). ##########
def get_foreground_background(image, mask):
    """Get foreground and background image by applying mask on image.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns foreground image, background image.
    """

    # 마스크를 이용해, 오리지날 컬러 이미지로부터 컬러 픽셀을 복사
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0: # instance_count 가 0보다 크다면
        # 이 함수에서는 컬러 스플래쉬가 목적이라서
        # 모든 인스턴스를 하나로 간주하기 때문에,
        # 인스턴스가 여러개여도 마스크는 하나의 레이어로 붕괴(통합)될 것
        # We're treating all instances as one, so collapse the mask into one layer
        fore_mask = (np.sum(mask, -1, keepdims=True) >= 1)
        back_mask = (np.sum(mask, -1, keepdims=True) < 1)

        foreground = np.where(fore_mask, image, 255).astype(np.uint8)
        background = np.where(back_mask, image, 0).astype(np.uint8)
    else:
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        foreground = gray.astype(np.uint8)
        background = gray.astype(np.uint8)
    return foreground, background

def crop_and_pad(image_in, image_out, bbox):
    """
    Crop image by bbox and pad with [255,255,255] to make it square.
    Save the image in image_out path.

    # input:
    image_in : numpy array
    image_out : output image path
    bbox : bounding box [y1, x1, y2, x2]
    # return:
    adjusted bbox (will be used in image rendering process)
    """
    if (bbox[2] - bbox[0]) % 2 != 0:
        bbox[0] += 1
    if (bbox[3] - bbox[1]) % 2 != 0:
        bbox[1] += 1

    y1, x1, y2, x2 = bbox
    crop_img = image_in[y1:y2, x1:x2]

    # roi adjust (square)
    w = x2 - x1
    h = y2 - y1

    if w >= h:
        p = int((w - h)/2)
        # cv2.copyMakeBorder(src, top, bottom, left, right, borderType)
        img_padding= cv2.copyMakeBorder(crop_img, p,p,0,0, cv2.BORDER_CONSTANT,value=[255,255, 255])
    else:
        p = int((h - w)/2)
        img_padding= cv2.copyMakeBorder(crop_img, 0,0,p,p, cv2.BORDER_CONSTANT,value=[255,255, 255])

    cv2.imwrite(image_out, cv2.cvtColor(img_padding, cv2.COLOR_RGB2BGR))
    return bbox

# 이미지에서 segmentation 적용하는 함수
def get_mask_save_segimage(model, image_path,
                            fore_file_path, back_file_path, save_back=True):
    """
    Return mask and save segmented image.
    # input
    model : model(See main.py. model is declared by modellib.MaskRCNN(...))
    image_path : input image path
    fore_file_path : foreground file path
    back_file_path : background file path
    save_back : boolean. whether save background image or not.
                usually set True on user image, and False on style image.
    # return
    mask and bounding box for the object
    mask : [H, W, N] instance binary masks
    bbox : [y1, x1, y2, x2]
    """
    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    # get foreground and background images
    fore, back = get_foreground_background(image, r['masks'])

    if save_back :
        # Save output background
        skimage.io.imsave(back_file_path, back)

    # crop and pad foreground image : to make it square
    # and save output in function
    bbox = crop_and_pad(fore, fore_file_path, r['rois'][0])

    print("Foreground Saved to ", fore_file_path)
    print("Background Saved to ", back_file_path)
    return r['masks'], bbox

def user_style_seg(user_input, style_input, model, weight, output_dir):
    """
    Apply segmentation on user image and style images.
    Save foreground and background images.
    # input
    user_input : user image path
    style_input : style image path
    model : model
    weight : weight path
    output_dir : output image directory
    # return
    user_fore: user_foreground image path
    user_back: user_background image path
    style_fore: style_foreground image path
    user_mask : [H, W, N] instance binary masks
    user_bbox : [y1, x1, y2, x2]
    """
    # filenames
    user_fore = output_dir+"user_foreground_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
    user_back = output_dir+"user_background_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
    style_fore = output_dir+"style_foreground_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
    style_back = output_dir+"style_background_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())

    # get mask and save segmented images (foreground, background)
    user_mask, user_bbox = get_mask_save_segimage(model, user_input,
                                        user_fore, user_back, save_back=True)
    _, _ = get_mask_save_segimage(model, style_input,
                                        style_fore, style_back, save_back=False)

    return user_fore, user_back, style_fore, user_mask, user_bbox


def image_rendering(tshirt, background, user_bbox, user_mask, output_dir):
    """
    image rendering : generated tshirt image on background image.
    tshirt image will be resized.

    # input
    tshirt : generated tshirt image path
    background : background image path
    user_bbox : user image bbox [y1, x1, y2, x2]
    user_mask : user mask from user image segmentation
    output_dir : output path to save final output image
    """
    # tshirt image size maybe 256x256
    #t = skimage.io.imread(tshirt)
    #bg = skimage.io.imread(background)
    t = cv2.imread(tshirt, cv2.IMREAD_COLOR)
    bg = cv2.imread(background, cv2.IMREAD_COLOR)
    bg_h, bg_w, _ = bg.shape

    """
    # 1. Resize, Crop and Pad generated tshirt image
    """
    y1, x1, y2, x2 = user_bbox
    # roi adjust (rect)
    w = x2 - x1
    h = y2 - y1

    if w >= h:
        p = int((w - h)/2)
        # 1) Resize tshirt image to original resolution
        #    because tshirt image had been resized when Cycle GAN done.
        t_resized = skimage.transform.resize(t, (w, w))
        # 2) Crop tshirt image to bbox size
        #    because tshirt image had been padded when segmentation done.
        t_crop = t_resized[p:p+h, :]
    else:
        p = int((h - w)/2)
        # 1) Resize tshirt image to original resolution
        #    because tshirt image had been resized when Cycle GAN done.
        t_resized = skimage.transform.resize(t, (h, h))
        # temp
        print('t_resized')
        print(t_resized.shape)
        # 2) Crop tshirt image to bbox size
        #    because tshirt image had been padded when segmentation done.
        t_crop = t_resized[:, p:p+w]
        # temp
        print('t_crop')
        print(t_crop.shape)


    # 3) Pad tshirt to original size
    #    because tshirt image had been cropped when segmentation done.
    # cv2.copyMakeBorder(src, top, bottom, left, ri ght, borderType)
    t_padding= cv2.copyMakeBorder(t_crop, y1,bg_h-y2,x1,bg_w-x2, cv2.BORDER_CONSTANT, value=[255,255,255])

    print('t_padding')
    print(t_padding.shape)
    print(t_padding)

    out_path = output_dir+"final_output_tpadding_0.jpg".format(datetime.datetime.now())
    #cv2.imwrite(out_path, cv2.cvtColor(t_padding, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_path, t_padding)


    print('original_user')
    print(bg.shape)


    """
    # 2. rendering
    """
    _, back = get_foreground_background(bg, user_mask)
    fore, _ = get_foreground_background(t_padding, user_mask)
    out_path = output_dir+"final_output_tpadding.jpg".format(datetime.datetime.now())
    cv2.imwrite(out_path, cv2.cvtColor(fore, cv2.COLOR_RGB2BGR))

    out = np.where(back==[0,0,0], [0,255,255], [0,0,255]).astype(np.uint8)

    """
    # 3. image save
    """
    out_path = output_dir+"final_output_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
