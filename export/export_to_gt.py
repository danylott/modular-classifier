import time
import os
from PIL import Image, ImageDraw

from get_data_from_db import get_data_from_db


def export_to_gt(input_classes, output_folder):
    class_list, image_list = get_data_from_db(input_classes)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    image_result = []
    label_result = []

    for idx, base in enumerate(image_list):
        image = Image.new('L', (base['width'], base['height']))
        d = ImageDraw.Draw(image)
        d.polygon(base['annotation'], fill=255)
        label_name = os.path.basename(os.path.splitext(base['path'])[0]) + '.png'
        image.save(output_folder + label_name)

        image_result.append(base['path'])
        label_result.append(label_name)

    print(f'Added {len(image_list)} images to gt dataset!')
    print(f'Added {len(class_list)} classes to gt dataset!')

    return image_result, label_result


if __name__ == "__main__":
    inp = ['TESTME']
    export_to_gt(inp, output_folder=f"ground_truth/{int(time.time())}/")
