import os
import json
import random
import imutils
from pprint import pprint
import numpy as np
import datetime
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='path to input folder')
parser.add_argument('--coco', help='path to coco')
parser.add_argument('--output', help='path to output folder')
parser.add_argument('--augmcount', help='count of images generated from each image')
parser.add_argument('--angle', help='max rotation angle')

args = parser.parse_args()

PATH_TO_FOLDER = args.input  # 'dataset/augmentation/test/'
PATH_TO_COCO = args.coco  # 'dataset/coco'
PATH_TO_OUTPUT = args.output  # 'dataset/output'
FILES_DESIRED_FOR_EACH = int(args.augmcount)  # 5
MAX_ANGLE = int(args.angle) # 25

NOISE = ["gauss", "s&p", "speckle"]


class CroppedStickerError(Exception):
    '''Sticker was cropped - impossible to create good segmentation'''


def noisy(noise_typ,image):
    """
    'gauss'     Gaussian-distributed additive noise.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


def random_noise(img):
    return noisy(random.choice(NOISE), img)


def transform_array_to_np(polygon):
    np_polygon = []
    for i in range(0, len(polygon), 2):
        np_polygon.append([polygon[i], polygon[i + 1]])
    
    np_polygon = np.array(np_polygon)
    return np_polygon

def transform_np_to_array(np_polygon):
    polygon = []
    
    for point in np_polygon:
        polygon.append(point[0])
        polygon.append(point[1])

    return polygon


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def rotate_polygon(polygon, origin, degrees, image_width, image_height):

    new_polygon = rotate(polygon, origin=origin, degrees=-degrees)
    new_polygon = [[int(x), int(y)] for x, y in new_polygon]
    for point in new_polygon:
        if point[0] <= 0 or point[0] >= image_width or point[1] <= 0 or point[1] >= image_height:
            raise CroppedStickerError()
            # print('bad sticker')

    return new_polygon


def random_rotation(img, image_shapes):
    angle = random.randint(0, 2 * MAX_ANGLE) - MAX_ANGLE
    rotated = imutils.rotate(img, angle)
    image_bbox, image_segmentation, _, image_width, image_height, _ = image_shapes
    image_bbox = np.array([[image_bbox[0], image_bbox[1]], [image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]]])

    origin = (image_width // 2, image_height // 2)

    rotated_bbox = rotate_polygon(image_bbox, origin, angle, image_width, image_height)
    rotated_segmentation = rotate_polygon(transform_array_to_np(image_segmentation), origin, angle, image_width, image_height)

    return (rotated, rotated_bbox, rotated_segmentation)


def calculate_bbox(segmentation):
    left_top = (10000, 10000)
    right_bottom = (0, 0)
    for i in range(0, len(segmentation), 2):
        left_top = (min(segmentation[i], left_top[0]), min(segmentation[i + 1], left_top[1]))
        right_bottom = (max(segmentation[i], right_bottom[0]), max(segmentation[i + 1], right_bottom[1]))

    # print(left_top, right_bottom)
    return [left_top[0], left_top[1], right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]]


def create_image_dict(idx, width, height, file_name):
    image_dict = {
        'id': idx,
        'width': width,
        'height': height,
        'file_name': file_name,
        'license': None,
        'flickr_url': "",
        'coco_url': None,
        'date_captured': str(datetime.datetime.now()),
    }
    return image_dict


def create_annotation_dict(id_annotation, id_image, category, segmentation, area, bbox):
    annotation_dict = {
        'id': id_annotation,
        'image_id': id_image,
        'category_id': category,
        'segmentation': [segmentation],
        'area': area,
        'bbox': bbox,
        "iscrowd": 0,
    }
    return annotation_dict


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
}

images_names = [el for el in os.listdir(PATH_TO_FOLDER) if el.endswith('.jpg') or el.endswith('.JPG') or el.endswith('.png') or el.endswith('.PNG') or el.endswith('.jpeg') or el.endswith('.JPEG')]

# images = [os.path.join(PATH_TO_FOLDER, f) for f in os.listdir(PATH_TO_FOLDER) if os.path.isfile(os.path.join(PATH_TO_FOLDER, f))]

num_generated_files = num_files_desired = 0

coco = None
with open(PATH_TO_COCO, 'r') as f:
    coco = json.load(f)
    print(coco.keys())
    images = coco['images'].copy()
    coco['images'] = []
    id_images = 0
    # print(images)
    image_ids = {image['id']: (image['file_name'], image['width'], image['height']) for image in images if image['file_name'] in images_names}
    pprint(image_ids)

    annotations = coco['annotations'].copy()
    coco['annotations'] = []
    id_annotations = 0

    image_shapes = {image_ids[el['image_id']][0]: (el['bbox'], el['segmentation'][0], el['category_id'], image_ids[el['image_id']][1], image_ids[el['image_id']][2], el['area']) for el in annotations if el['image_id'] in image_ids}
    pprint(image_shapes)

    for image_name in images_names:
        image_bbox, image_segmentation, image_category, image_width, image_height, image_area = image_shapes[image_name]
        img = cv2.imread(PATH_TO_FOLDER + image_name)

        image_dict = create_image_dict(id_images, image_width, image_height, image_name)
        annotation_dict = create_annotation_dict(id_annotations, id_images, image_category, image_segmentation, image_area, image_bbox)

        id_images += 1
        id_annotations += 1
        print('Image:', image_dict)
        print('Annotation:', annotation_dict)

        coco['images'].append(image_dict)
        coco['annotations'].append(annotation_dict)

        num_files_desired += FILES_DESIRED_FOR_EACH
        while num_generated_files <= num_files_desired:
            num_transformations_to_apply = random.randint(1, len(available_transformations) + 1)

            num_transformations = 0
            is_rotated = False
            must_rotate = False
            new_img = img.copy()
            while num_transformations <= num_transformations_to_apply:
                try:
                    # random transformation to apply for a single image
                    key = random.choice(list(available_transformations))
                    if (key == 'rotate' and not is_rotated) or must_rotate:
                        new_img, rotated_bbox, rotated_segmentation = random_rotation(new_img, image_shapes[image_name])

                        rotated_segmentation = transform_np_to_array(rotated_segmentation)
                        rotated_bbox = transform_np_to_array(rotated_bbox)
                        rotated_bbox = calculate_bbox(rotated_segmentation)
                        
                        # use this to check bbox and segmentation on new images - will throw an error - use only for test
                        # rotated_bbox = np.array([[rotated_bbox[0], rotated_bbox[1]], [rotated_bbox[0] + rotated_bbox[2], rotated_bbox[1] + rotated_bbox[3]]])
                        # rotated_bbox = rotated_bbox.reshape((-1,1,2))
                        # cv2.polylines(new_img, [rotated_bbox], True, (0,255,255))
                        # rotated_segmentation = transform_array_to_np(rotated_segmentation)
                        # rotated_segmentation = rotated_segmentation.reshape((-1,1,2))
                        # cv2.polylines(new_img, [rotated_segmentation], True, (0,255,255))
                        
                        is_rotated = True
                        must_rotate = False
                    else:
                        new_img = random_noise(new_img)

                    num_transformations += 1

                except CroppedStickerError:
                    print('Sticker was cropped - impossible to create good segmentation - continue')
                    must_rotate = True
                    new_img = img.copy()
                    continue

                # except ValueError:
                #     print('ValueError for this augmentation - continue ')
                #     continue
            
            if not is_rotated:
                        rotated_bbox, rotated_segmentation = image_bbox, image_segmentation

            
            new_file_name = f'augmented_image_{num_generated_files}.jpg'
            new_file_path = PATH_TO_FOLDER + new_file_name

            print('Original Image:', image_bbox, image_segmentation)
            print('Rotated Image:', is_rotated, rotated_bbox, rotated_segmentation)

            # create records to write 
            image_dict = create_image_dict(id_images, image_width, image_height, new_file_name)
            annotation_dict = create_annotation_dict(id_annotations, id_images, image_category, rotated_segmentation, image_area, rotated_bbox)

            id_images += 1
            id_annotations += 1
            print('Image:', image_dict)
            print('Annotation:', annotation_dict)

            coco['images'].append(image_dict)
            coco['annotations'].append(annotation_dict)

            # write image to the disk
            cv2.imwrite(new_file_path, new_img)
                    
            num_generated_files += 1

with open(PATH_TO_OUTPUT, 'w') as f:
    json.dump(coco, f)

print('SUCCESS! -', num_generated_files, 'images have been created')
print('Output file: ' + PATH_TO_OUTPUT)
