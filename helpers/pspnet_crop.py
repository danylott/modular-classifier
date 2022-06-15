from keras_applications import imagenet_utils
from PIL import Image
import numpy as np
import cv2
import os

from helpers.builder import builder


offset = 5


def get_cropper(model_path, crop_height=384, crop_width=480, num_classes=2):

    # build the model
    net, base_model = builder(num_classes, (crop_height, crop_width), 'ResNet50')

    # load weights
    print('Loading the weights...')
    net.load_weights(model_path)
    return net


def cut_sticker(img):
    (h, w) = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    return img[y + offset : y + h - offset, x + offset : x + w - offset]


def decode_one_hot(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)


def load_image(name):
    img = Image.open(name)
    return np.array(img)


def pspnet_crop(net, input_path, output_path, crop_height=384, crop_width=480):
    # check the image path
    if not os.path.exists(input_path):
        raise ValueError(f"The path {input_path} does not exist the image file")

    image = cv2.resize(load_image(input_path),
                       dsize=(crop_width, crop_height))
    image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

    # image processing
    if np.ndim(image) == 3:
        image = np.expand_dims(image, axis=0)
    assert np.ndim(image) == 4

    # get the prediction
    prediction = net.predict(image)

    if np.ndim(prediction) == 4:
        prediction = np.squeeze(prediction, axis=0)

    # decode one-hot


    img_masked = cv2.imread(input_path)

    prediction = decode_one_hot(prediction)
    prediction = cv2.resize(np.uint8(prediction),
                            dsize=(img_masked.shape[1], img_masked.shape[0]), interpolation=cv2.INTER_CUBIC)

    x, y, w, h = cv2.boundingRect(prediction)
    # print(x, y, w, h)
    # print(prediction)
    prediction = np.array(prediction, dtype=bool)
    # print(prediction.shape)
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    img_masked[~prediction, :] = [0, 0, 0]
    grey = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(grey > 0))
    hgt_rot_angle = cv2.minAreaRect(coords)[-1]
    angle = hgt_rot_angle + 90 if hgt_rot_angle < -45 else hgt_rot_angle
    (h, w) = img_masked.shape[0:2]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    corrected_image = cv2.warpAffine(
        img_masked,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    crop = cut_sticker(corrected_image)
    return crop


if __name__ == "__main__":
    pspnet_crop("dataset/images/IMG_0430r.JPG", "models/PSPNET_RESNET50_99mean.h5", 'dataset/1.png', 384, 480, 2)
