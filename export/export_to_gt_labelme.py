import time
import os

import labelme
from PIL import Image, ImageDraw

from get_data_from_db import get_data_from_db


def convert_annotation(annotation):
    result = []
    for i in range(0, len(annotation), 2):
        result.append([annotation[i], annotation[i + 1]])

    return result


def export_to_gt(input_classes, output_folder):
    class_list, image_list = get_data_from_db(input_classes)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    image_result = []
    label_result = []

    labels_txt = {'__ignore__': -1, '_background_': 0}
    index = 1
    for idx, base in enumerate(image_list):
        if str(base['cls']) not in labels_txt:
            labels_txt[str(base['cls'])] = index
            index += 1

    for idx, base in enumerate(image_list):
        shape = (base['height'], base['width'], 3)
        # TODO: fix str(base['cls'])
        label = [{'label': str(base['cls']), 'points': convert_annotation(base['annotation']),
                     'shape_type': 'polygon', 'flags': {},
                     'group_id': None,
                     'other_data': {'line_color': None, 'fill_color': None}}]

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=shape,
            shapes=label,
            label_name_to_value=labels_txt,
        )
        label_name = os.path.basename(os.path.splitext(base['path'])[0]) + '.png'
        labelme.utils.lblsave(output_folder + label_name, lbl)
        image_result.append(base['path'])
        label_result.append(label_name)

    print(f'Added {len(image_list)} images to gt dataset!')
    print(f'Added {len(class_list)} classes to gt dataset!')

    return image_result, label_result


if __name__ == "__main__":
    inp = ['auto_sem_seg']
    export_to_gt(inp, output_folder=f"ground_truth/{int(time.time())}/")
