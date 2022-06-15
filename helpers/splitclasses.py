import json

def split_classes(classes, input_file_coco, output_file_coco):
    class_id = 0
    image_id = 0
    coco = {}
    with open(input_file_coco) as f:
        coco = json.load(f)
        categories = coco['categories']
        categories = [category for category in categories if category['name'] in classes]
        id_categories = [category['id'] for category in categories]

        coco['categories'] = categories
        # print(categories)

        annotations = coco['annotations']
        annotations = [ann for ann in annotations if ann['category_id'] in id_categories]
        id_images = [ann['image_id'] for ann in annotations]

        coco['annotations'] = annotations
        # print(annotations)

        images = coco['images']
        images = [image for image in images if image['id'] in id_images]

        coco['images'] = images
        print(f"Successfully splitted: {len(coco['images'])} images!")
        # print(images)

    with open(output_file_coco, 'w') as f:
        json.dump(coco, f)

    return output_file_coco


if __name__ == '__main__':
    # with open('dataset/nike.json', 'w') as f:
    output = split_classes(['nike'], 'dataset/krack_71.json', 'dataset/nike.json')
        
