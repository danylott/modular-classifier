import time
from export_to_gt_labelme import export_to_gt
from split_gt import split_gt


def use_gt_from_classes(class_id_list, output_gt, input_im, output):
    images, labels = export_to_gt(class_id_list, output_gt)
    split_gt(input_im, output_gt, output, images, labels)


if __name__ == "__main__":
    classes = ['auto_sem_seg']
    output_ground_truth = f"ground_truth/{int(time.time())}/"
    input_images = '../back-end/images/'
    output_folder = f'semantic_segmentation/{int(time.time())}'
    use_gt_from_classes(classes, output_ground_truth, input_images, output_folder)
