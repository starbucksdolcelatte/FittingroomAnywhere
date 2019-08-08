"""
Mask R-CNN
Train on the toy Tshirt dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Tshirt.py train --dataset=/path/to/Tshirt/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 Tshirt.py train --dataset=/path/to/Tshirt/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Tshirt.py train --dataset=/path/to/Tshirt/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Tshirt.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Tshirt.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/drive/My Drive/Colab Notebooks/Mask_RCNN")

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
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

# BalloonDataset -> TshirtDataset
class TshirtDataset(utils.Dataset):

    def load_Tshirt(self, dataset_dir, subset):
        """Load a subset of the Tshirt dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("Tshirt", 1, "Tshirt")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        # 여기에 있는 이 "via_region_data.json" 파일은 데이타셋이 있는 폴더의 train폴더, val폴더 각각에 이미지와 함께 들어 있음.
        # 우리도 우리의 데이터셋에 맞게 이 json 파일을 만들어야 한다.
        # VGG image annotation을 사용했는데 우리 것은 COCO 형식이다.
        # 어떻게 해야 할까?
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() 함수는 polygon을 mask로 변환하기 위해 이미지 사이즈를 필요로 한다.
            # VIA는 이 정보를 JSON에 포함하지 않았기 때문에,
            # 우리는 아래 코드에서 이미지 사이즈를 읽어 온다.
            # 데이타셋이 작을 때만 가능하다.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # 이제 TshirtDataset() 클래스에 아래 add_image() 를 통해 이미지를 추가할 것이다.
            # 여기에는 아래와 같은 정보가 저장된다.
            self.add_image(
                source="Tshirt", # source
                image_id=a['filename'],  #  : filename은 고유한 이름으로 사용해라.
                path=image_path, # 이미지 경로
                width=width, height=height, # 이미지 사이즈
                polygons=polygons) # polygon은 Json 파일에서 "shape_attributes"에 해당하는 것으로, name, all_points_x, all_points_y가 있다.



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


# model을 트레이닝 하는 함수
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TshirtDataset()
    dataset_train.load_Tshirt(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TshirtDataset()
    dataset_val.load_Tshirt(args.dataset, "val")
    dataset_val.prepare()

    # *** 이 트레이닝 스케줄은 예시입니다. 필요에 따라 업데이트하세요 *** #
    # 우리는 아주 작은 데이터셋을 사용하고, COCO trained weight로부터 시작하기 때문에,
    # 트레이닝을 너무 오래 할 필요는 없습니다.
    # 또한, 모든 레이어를 트레이닝 할 필요도 없고, head 부분만 트레이닝하면 됩니다.
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')



# 컬러 스플레쉬 효과를 사진에 적용하는 함수
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # 먼저 흑백 사진 카피를 만듭니다.
    # 이 흑백사진은 여전히 3 RGB 채널을 가지고 있지만요.
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # 마스크를 이용해, 오리지날 컬러 이미지로부터 컬러 픽셀을 복사
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # 이 함수에서는 컬러 스플래쉬가 목적이라서
        # 모든 인스턴스를 하나로 간주하기 때문에,
        # 인스턴스가 여러개여도 마스크는 하나의 레이어로 붕괴(통합)될 것
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


# 이미지 또는 비디오에서 color splash를 적용하는 함수
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Tshirt.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Tshirt/dataset/",
                        help='Directory of the Tshirt dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        # 위에서 선언한 TshirtConfig 인스턴스 생성하여 변수 등등을 설정
        config = TshirtConfig()
    else:
        # TshirtConfig 클래스를 상속받는 InferenceConfig 라는 클래스를 선언한다.
        class InferenceConfig(TshirtConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    #############################
    # Create model
    # 모델을 생성하는 부분!!
    #############################
    if args.command == "train":
        # mode 는 training 모드
        # config 는 TshirtConfig 또는 InferenceConfig 인스턴스 객체
        # model_dir 는 로그를 저장하는 디렉토리. 여기에 epoch당 웨이트 파일이 저장됨
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
