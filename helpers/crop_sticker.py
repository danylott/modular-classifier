import cv2
import numpy as np
from skimage import draw
import numpy
from PIL import Image, ImageDraw


offset = 5


def cut_sticker(img):
    (h, w) = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    return img[y + offset : y + h - offset, x + offset : x + w - offset]


def crop_sticker(input, output, annotation):
    annotation_copy = []
    for i in range(0, len(annotation), 2):
        annotation_copy.append((annotation[i + 1], annotation[i]))

    annotation = np.array(annotation_copy)

    img_masked = cv2.imread(input)
    mask = draw.polygon2mask((img_masked.shape[0], img_masked.shape[1]), annotation)
    x, y, w, h = cv2.boundingRect(annotation)
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    img_masked[~mask, :] = [0, 0, 0]
    print(mask)
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
    cv2.imwrite(output, crop)

    return output


if __name__ == "__main__":

    annotation = [
        140,
        677,
        961,
        516,
        1062,
        1033,
        279,
        1197,
    ]

    crop_sticker('dataset/images/IMG_0430r.JPG', 'out.png', annotation)


