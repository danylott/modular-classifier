import os
from shutil import copyfile


def create_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_images_labels(path):
    create_not_exists(path)
    create_not_exists(os.path.join(path, 'images'))
    create_not_exists(os.path.join(path, 'labels'))


def create_folders(path):
    create_not_exists(path)
    create_images_labels(os.path.join(path, 'train'))
    create_images_labels(os.path.join(path, 'test'))
    create_images_labels(os.path.join(path, 'valid'))


def copy_image_label(image_path, label_path, path):
    copyfile(image_path, os.path.join(path, 'images', os.path.basename(image_path)))
    copyfile(label_path, os.path.join(path, 'labels', os.path.basename(label_path)))


def split_gt(path_to_images, path_to_labels, path_to_output, image_names, label_names):
    create_folders(path_to_output)
    count_train = 0
    count_test = 0
    count_valid = 0

    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(path_to_images, image_name)
        label_path = os.path.join(path_to_labels, os.path.basename(os.path.splitext(image_name)[0]) + '.png')
        assert label_path == os.path.join(path_to_labels, label_names[idx])
        if os.path.exists(image_path):
            if os.path.exists(label_path):
                if idx % 5 == 0:
                    copy_image_label(image_path, label_path, os.path.join(path_to_output, 'test'))
                    count_test += 1
                elif (idx - 1) % 5 == 0:
                    copy_image_label(image_path, label_path, os.path.join(path_to_output, 'valid'))
                    count_valid += 1
                else:
                    copy_image_label(image_path, label_path, os.path.join(path_to_output, 'train'))
                    count_train += 1

    print(f'Saved {count_train} train images!')
    print(f'Saved {count_test} test images!')
    print(f'Saved {count_valid} valid images!')


if __name__ == "__main__":
    import time
    images = ['classes/image1598439407176.jpg', 'classes/image1598439407280.jpg']
    labels = ['image1598439407176.png', 'image1598439407280.png']
    split_gt('../../back-end/images/', 'ground_truth/1598514412', f'semantic_segmentation/{int(time.time())}',
             images, labels)
