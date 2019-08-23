import Tshirt_v5 as ts
import os
import sys
import cv2
import numpy as np
"""
MRCNN_WEIGHT = 'path/to/mrcnn/weight'
INPUT_USER_IMAGE = 'path/to/user/image'
INPUT_STYLE_IMAGE = 'path/to/style/image'
OUTPUT_DIR = 'path/to/output/image/dir/'
ROOT_DIR = 'path/to/output/Mask_RCNN/dir/'
"""
ROOT_DIR = 'C:/Users/chief/FittingroomAnywhere/segmentation/Mask_RCNN/'
MRCNN_WEIGHT = ROOT_DIR + 'logs/mask_rcnn_tshirt_0028.h5'
INPUT_USER_IMAGE = ROOT_DIR + 'samples/Tshirt/img/input/user.jpg'
INPUT_STYLE_IMAGE = ROOT_DIR + 'samples/Tshirt/img/input/style.jpg'
OUTPUT_DIR = ROOT_DIR + 'samples/Tshirt/img/output/'

# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/chief/FittingroomAnywhere/segmentation/Mask_RCNN")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

def main():
    '''
    # TshirtConfig 클래스를 상속받는 InferenceConfig 라는 클래스를 선언한다.
    class InferenceConfig(ts.TshirtConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    model.load_weights(MRCNN_WEIGHT, by_name=True)
    user_fore, user_back, style_fore, user_mask, user_bbox = ts.user_style_seg(INPUT_USER_IMAGE, INPUT_STYLE_IMAGE, model, MRCNN_WEIGHT, OUTPUT_DIR)

    # temp function; cannot be used. just usage example
    #generated_tshirt = CycleGAN(user_fore, style_fore, weight, ...)

    np.save(OUTPUT_DIR+'user_bbox.npy', user_bbox) # .npy extension is added if not given
    np.save(OUTPUT_DIR+'user_mask.npy', user_mask) # .npy extension is added if not given
    '''
    user_bbox = np.load(OUTPUT_DIR+'user_bbox.npy')
    user_mask = np.load(OUTPUT_DIR+'user_mask.npy')

    # You can use this function.
    user_back = OUTPUT_DIR+'user_background_20190823T185033.jpg'
    #user_bbox = [182, 129, 536, 449]
    ts.image_rendering(OUTPUT_DIR+'fakeA.jpg', user_back, user_bbox, user_mask, OUTPUT_DIR)


if __name__ == '__main__':
    main()
