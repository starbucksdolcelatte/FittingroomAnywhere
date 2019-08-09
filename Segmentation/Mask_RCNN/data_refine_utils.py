import os
import json
import shutil
import numpy as np

########################################
# Image copy&paste by using JSON file
########################################
def img_copy_by_json(from_path, to_path, json_file):
    """
    JSON 파일 안에 나열된 이미지파일들을
    from_path에서 to_path로 copy & paste 한다.
    -----------------------------------------
    from_path : source image directory path
    to_path : destination image directory path
    json_file : 이미지 파일명을 가지고 있는 json 파일 path
    Json file 형식은 {"images":[filename1, filename2, ...]}
    """
    if os.path.exists(from_path) == False:
        print('from_path not exists')
        return
    if os.path.exists(to_path) == False:
        os.mkdir(to_path)

    if from_path[-1] != '/':
        from_path += '/'
    if to_path[-1] != '/':
        to_path += '/'

    # images 파일명이 있는 json 파일 읽어들이기
    with open(json_file) as json_file:
        json_data = json.load(json_file)

    # image 파일 붙여넣기
    for data in json_data["images"]:
        shutil.copy(from_path + data, to_path + data)



def img_filenames_to_json(dir_path, json_file):
    """
    dir_path 안에 있는 모든 이미지 파일명들을 담은 JSON 파일을 생성
    -----------------------------------------
    dir_path : path/to/directory
    json_file : path/to/json/file/to/create
    (e.g. json_file = 'mydir/new_json.json')
    """
    if os.path.exists(dir_path) == False:
        print('dir_path not exists')
        return

    # dir_path 있는 anotation 파일 이름 추출
    img_files = os.listdir(dir_path)

    # JSON 파일로 저장
    json_dict = {"images":img_files}
    with open(json_file, 'w') as f:
        json.dump(json_dict, f)



##############################################
# Annotation copy&paste according image files
##############################################
def anno_copy_by_img(img_path, from_path_anno, to_path_anno):
    """
    img_path 안의 이미지에 해당하는 annotation 파일을
    from_path_anno에서 to_path_anno로 copy & paste한다.
    -----------------------------------------
    img_path : path/to/DeepFashion2_selected_images
    from_path_anno : source of DeepFashion2_annotation
    to_path_anno : destination of DeepFashion2_annotation
    """
    if os.path.exists(from_path_anno) == False:
        print('from_path_anno not exists')
        return
    if os.path.exists(img_path) == False:
        print('img_path not exists')
        return
    if os.path.exists(to_path_anno) == False:
        os.mkdir(to_path_anno)

    if from_path_anno[-1] != '/':
        from_path_anno += '/'
    if to_path_anno[-1] != '/':
        to_path_anno += '/'

    # 이미지 파일명을 리스트로 받아와서 저장
    img_data = os.listdir(img_path)
    json_data = []

    for img_file_name in img_data:
        json_data.append(img_file_name.replace(".jpg", ".json"))

    for json_file_name in json_data:
        shutil.copy(from_path_anno + json_file_name,
                    to_path_anno + json_file_name)




########################################
# Extract specified category images
########################################
def is_not_contains_category(anno_dir, category_id):
    """
    DeepFashion2의 annotation 파일들이 있는 디렉토리에서
    최대 item2까지 확인하여 category_id를 포함하지
    않는 파일명을 출력 및 리스트로 리턴한다.
    -----------------------------------------
    anno_dir : path/to/DeepFashion2_annotation
    category_id(integer) : DeepFashion2 category id below
        1 : 'short sleeve top'
        2 : 'long sleeve top'
        3 : 'short sleeve outwear'
        4 : 'long sleeve outwear'
        5 : 'vest'
        6 : 'sling'
        7 : 'shorts'
        8 : 'trousers'
        9 : 'skirt'
        10 : 'short sleeve dress'
        11 : 'long sleeve dress'
        12 : 'vest dress'
        13 : 'sling dress'
    """
    not_contains = []

    if os.path.exists(anno_dir) == False:
        print('anno_dir not exists')
        return
    if anno_dir[-1] != '/':
        anno_dir += '/'

    # anno_dir에 있는 anotation 파일명 추출
    directory = os.listdir(anno_dir)

    for anno in directory:
        # annotation.json 파일 읽어들이기
        with open(anno_dir + anno) as json_file:
            json_data = json.load(json_file)

        # item1의 카테고리가 category_id가 아니라면,
        if json_data['item1']['category_id']!=category_id:
            try:
                # item2의 카테고리가 category_id인지 보자.
                # 만약 아니라면, 파일명을 print
                if(json_data['item2']['category_id']!=category_id):
                    print('Not contains short sleeve top : ', anno)
                    not_contains.append(anno)
                else:
                    print(anno, '---- item 2 ----', json_data['item2']['category_name'])

            except:
                print('Not contains short sleeve top : ',anno)
                not_contains.append(anno)
        else:
            print(anno, '---- item 1 ----', json_data['item1']['category_name'])
    # End of for.
    return not_contains


########################################
# Modify Annotation
########################################
def seg_to_points(segmentation):
    """
    segmentation을 all_points_x, all_points_y로 변환
    """
    hl = len(segmentation)//2
    x = [segmentation[x*2] for x in range(hl)]
    y = [segmentation[x*2+1] for x in range(hl)]
    return x, y

def lm_to_points(landmarks):
    """
    landmarks를 all_points_x, all_points_y로 변환
    """
    hl = len(landmarks)//3
    x = [landmarks[x*3] for x in range(hl)]
    y = [landmarks[1+x*3] for x in range(hl)]
    return x, y

def COCO_to_VIA(anno_dir, image_dir, category_id, save_anno_dir, mode="segmentation"):
    """
    DeepFashion2의 COCO anno 를 Mask R-CNN의 VIA anno로 변환.
    특정 카테고리의 하나의 인스턴스에 대해서만 작동함.
    anno_dir : DeepFashion2의 annotation 파일이 있는 directory
    image_dir : DeepFashion2의 image 파일이 있는 directory
    category_id : 추출하고자 하는 category_id
    save_anno_dir : 변환한 annotation을 저장할 directory
    mode : landmarks or segmentation.
    """
    import skimage

    if os.path.exists(anno_dir) == False:
        print('anno_dir not exists')
        return
    if os.path.exists(image_dir) == False:
        print('image_dir not exists')
        return
    if os.path.exists(save_anno_dir) == False:
        os.mkdir(save_anno_dir)

    if anno_dir[-1] != '/':
        anno_dir += '/'
    if image_dir[-1] != '/':
        image_dir += '/'
    if save_anno_dir[-1] != '/':
        save_anno_dir += '/'

    # anno_dir에 있는 anotation 파일 이름 추출
    directory = os.listdir(anno_dir)

    # VIA format annotation이 저장될 dict
    VIA_dict = {}

    # VIA format annotation이 저장될 JSON file
    # 파일이름은 반드시 아래 이름으로 해야 함
    json_file_name = 'via_region_data.json'

    for anno in directory:
        # annotation.json 파일 읽어들이기
        with open(anno_dir + anno) as json_file:
            json_data = json.load(json_file)

        # image file name
        img_path = anno.replace('.json', '.jpg')

        # img_size config
        img_abs_path = os.path.join(image_dir, img_path)
        image = skimage.io.imread(img_abs_path)
        height, width = image.shape[:2]
        img_size = height*width

        # item1의 카테고리가 category_id이 아니라면,
        if json_data['item1']['category_id']!=category_id:
            try:
                # item2의 카테고리가 category_id인지 보자.
                if(json_data['item2']['category_id']!=category_id):
                    print('Not contains short sleeve top : ', anno)
                else:
                    if mode == "segmentation":
                        all_points_x, all_points_y = seg_to_points(json_data['item2']['segmentation'][0])
                    elif mode == "landmarks":
                        all_points_x, all_points_y = lm_to_points(json_data['item2']['landmarks'])
                    VIA_dict[img_path] = {
                        "fileref": "",
                        "size": img_size,
                        "filename": img_path,
                        "base64_img_data": "",
                        "file_attributes": {},
                        "regions": {
                          "0": {
                            "shape_attributes": {
                              "name": "polygon",
                              "all_points_x": all_points_x,
                              "all_points_y": all_points_y
                            },
                            "region_attributes": {}
                          }
                        }
                    }

            except:
                print('Not contains short sleeve top : ',anno)
        else:
            if mode == "segmentation":
                all_points_x, all_points_y = seg_to_points(json_data['item1']['segmentation'][0])
            elif mode == "landmarks":
                all_points_x, all_points_y = lm_to_points(json_data['item1']['landmarks'])
            VIA_dict[img_path] = {
                "fileref": "",
                "size": img_size,
                "filename": img_path,
                "base64_img_data": "",
                "file_attributes": {},
                "regions": {
                  "0": {
                    "shape_attributes": {
                      "name": "polygon",
                      "all_points_x": all_points_x,
                      "all_points_y": all_points_y
                    },
                    "region_attributes": {}
                  }
                }
            }
        # End of if.
    # End of for.
    with open((save_anno_dir + json_file_name), 'w') as f:
        json.dump(VIA_dict, f)
