import datetime
import json

from export.get_data_from_db import get_data_from_db


def polygon_area(x_list, y_list):
    area = 0.0
    n = len(x_list)

    j = n - 1
    for i in range(0, n):
        area += (x_list[j] + x_list[i]) * (y_list[j] - y_list[i])
        j = i

    return int(abs(area / 2.0))


def calculate_bbox(annotation):
    x_list = annotation[0:][::2]
    y_list = annotation[1:][::2]
    return [min(x_list), min(y_list), max(x_list) - min(x_list), max(y_list) - min(y_list)]


def export_to_coco(input_classes, output_json, base_path=""):
    class_list, image_list = get_data_from_db(input_classes)

    # create result dict
    coco = {'info': {
        "year": str(datetime.datetime.now().year),
        "version": "1.0",
        "description": "Krack labels",
        "contributor": "",
        "url": "",
        "date_created": str(datetime.datetime.now())
    }}

    images = []
    annotations = []
    important_keys_list = ('_id', 'width', 'height', 'path', 'cls', 'annotation')
    for idx, image in enumerate(image_list):
        if all(key in image for key in important_keys_list):
            images.append({
                "id": str(image['_id']),
                "width": image['width'],
                "height": image['height'],
                "file_name": base_path + image['path'],
                "flickr_url": "",
                "coco_url": None,
                "date_captured": str(datetime.datetime.now())
            })
            annotations.append({
                "id": idx,
                "image_id": str(image['_id']),
                "category_id": str(image['cls']),
                "segmentation": [image['annotation']],
                "area": polygon_area(image['annotation'][0:][::2], image['annotation'][1:][::2]),
                "bbox": calculate_bbox(image['annotation']),
                "iscrowd": 0,
            })

    categories = []
    for cls in class_list:
        categories.append({
            "id": str(cls['_id']),
            "name": cls['name'],
            "supercategory": "object",
        })

    coco['images'] = images
    coco['annotations'] = annotations
    coco['categories'] = categories
    coco['licenses'] = []

    # print(coco)
    print(f'Added {len(images)} images to coco dataset!')
    print(f'Added {len(annotations)} annotations to coco dataset!')
    print(f'Added {len(categories)} classes to coco dataset!')

    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f'Coco json saved at: {output_json}')
    return coco


if __name__ == "__main__":
    base_image_path = "http://api.krack.springsapps.com/images/"
    inp = ['TESTME']
    export_to_coco(inp, 'coco_auto.json', base_image_path)
