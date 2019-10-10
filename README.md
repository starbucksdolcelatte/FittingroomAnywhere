# FittingroomAnywhere
Put on clothes anywhere and check how cool you are!<br>
: A virtual fitting room service for online shopping

# project folder overview
PROJECT_DIR = "."
- main.py
- mrcnn
  - tshirt.py
  - other .py files
  - logs (mrcnn saved model)
    - mrcnn_tshirt.h5 (weight)
- cyclegan
  - model.py
  - load_data.py
  - other .py files
- datasets
  - white2stripe
    - input
      - user_input.jpg / .png
    - segmented
      - segmented.png
    - fake_output (gan generated output)
      - segmented_fake.png
    - output (final rendered output)
      - output.png / .jpg
    - model (cyclegan saved model)
      - G_A2B.json (model)
      - G_A2B.hdf5 (weight)


# GAN
Cycle GAN repository for image genereation

# Segmentation
Mask R-CNN repository for image segmentation (adopted from matterport/MaskRCNN)

# presentation
Presentation notes for this project. We had presented this project for 3 times, and the final presentation was on Aug 27, 2019.

- [190730](https://github.com/starbucksdolcelatte/FittingroomAnywhere/blob/master/Presentation/190730-presentation.md)

- [190820](https://github.com/starbucksdolcelatte/FittingroomAnywhere/blob/master/Presentation/190820-presentation.md)

- 190827(to be updated)
