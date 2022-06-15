from shapely.geometry import Polygon
import shutil
import os
from PIL import Image

from craft.test import get_bboxes
from ocr.demo import get_recognitions
import config.config as config


threshold = config.TEXT_MARKUP_THRESHOLD


class MaximalIntersection:
    def __init__(self):
        self._model = None
        self._color = None
        self._size = None

    def set_handler(self, field, intersection_percent, bbox):
        if field == 'Model':
            self.set_model(intersection_percent, bbox)
        elif field == 'Color':
            self.set_color(intersection_percent, bbox)
        elif field == 'Size':
            self.set_size(intersection_percent, bbox)

    def set_model(self, intersection_percent, model_bbox):
        if self._model is None or intersection_percent > self._model[0]:
            self._model = (intersection_percent, model_bbox)

    def get_model(self):
        if self._model is None:
            return []
        return [format_bbox(self._model[1])]

    def set_color(self, intersection_percent, color_bbox):
        if self._color is None or intersection_percent > self._color[0]:
            self._color = (intersection_percent, color_bbox)

    def get_color(self):
        if self._color is None:
            return []
        return [format_bbox(self._color[1])]

    def set_size(self, intersection_percent, size_bbox):
        if self._size is None or intersection_percent > self._size[0]:
            self._size = (intersection_percent, size_bbox)

    def get_size(self):
        if self._size is None:
            return []
        return [format_bbox(self._size[1])]


def crop(image, bbox, path_to_save):
    cropped = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    cropped.save(path_to_save)


def choose_word_area(word_markup, text_markup, max_intersection_object):
    """Choose, if word corresponds to Model, Size, Color, or None (depends on threshold)"""
    word_polygon = Polygon(word_markup)
    max_intersection = 0
    type_max_intersection = None
    for markup in text_markup:
        if markup['polygon'].intersects(word_polygon):
            area = markup['polygon'].intersection(word_polygon).area
            intersection = area / word_polygon.area
            max_intersection_object.set_handler(markup['field'], intersection, word_markup)
            if intersection > max_intersection and intersection > threshold:
                type_max_intersection = markup['field']
                max_intersection = area

    return type_max_intersection


def prepare_text_markup(image, text_markup):
    width, height = image.size
    return [{
        'field': markup['field'],
        'polygon': Polygon([
            (markup['x'] * width, markup['y'] * height),
            ((markup['x'] + markup['w']) * width, markup['y'] * height),
            ((markup['x'] + markup['w']) * width, (markup['y'] + markup['h']) * height),
            (markup['x'] * width, (markup['y'] + markup['h']) * height),
        ]),
    }
        for markup in text_markup
    ]


def format_bbox(bbox):
    left_top_x = int(min([coord[0] for coord in bbox]))
    left_top_y = int(min([coord[1] for coord in bbox]))
    right_bottom_x = int(max([coord[0] for coord in bbox]))
    right_bottom_y = int(max([coord[1] for coord in bbox]))
    result = (left_top_x, left_top_y, right_bottom_x - left_top_x, right_bottom_y - left_top_y)
    return result


def sort_bboxes(contours):
    max_height = max([cnt[3] for cnt in contours])
    min_y = min([cnt[1] for cnt in contours])
    nearest = max_height * config.ONE_LEVEL_TEXT_COEF

    contours.sort(key=lambda rectangle: [int(nearest * round(float((rectangle[1] - min_y) - rectangle[3] / 2) / nearest)), rectangle[0]])


def get_ocr_bboxes(image_path, craft_net):
    bboxes = get_bboxes(image_path, craft_net, config.USE_GPU)
    try:
        bboxes.tolist()
    except AttributeError as e:
        print(f"error while transforming to list: {e}")
    
    return bboxes


def ocr(image_path, text_markup, craft_net, ocr_model, ocr_converter, ocr_opt):
    bboxes = get_ocr_bboxes(image_path, craft_net)
    max_intersection_object = MaximalIntersection()

    image = Image.open(image_path)
    text_markup = prepare_text_markup(image, text_markup)
    model_bboxes = []
    color_bboxes = []
    size_bboxes = []
    for bbox in bboxes:
        field = choose_word_area(bbox, text_markup, max_intersection_object)
        if field is not None:
            bbox = format_bbox(bbox)
            if field == 'Model':
                model_bboxes.append(bbox)
            elif field == 'Color':
                color_bboxes.append(bbox)
            elif field == 'Size':
                size_bboxes.append(bbox)

    try:
        shutil.rmtree('images/model/')
        shutil.rmtree('images/size/')
        shutil.rmtree('images/color/')
    except Exception as e:
        print(f"Error while creating/deleting folders: {e}")

    os.mkdir('images/model/')
    os.mkdir('images/size/')
    os.mkdir('images/color/')
    if model_bboxes:
        sort_bboxes(model_bboxes)
    else:
        model_bboxes = max_intersection_object.get_model()
    if color_bboxes:
        sort_bboxes(color_bboxes)
    else:
        color_bboxes = max_intersection_object.get_color()
    if size_bboxes:
        sort_bboxes(size_bboxes)
    else:
        size_bboxes = max_intersection_object.get_size()

    for idx, bbox in enumerate(model_bboxes):
        crop(image, bbox, f'images/model/{idx}.png')

    for idx, bbox in enumerate(size_bboxes):
        crop(image, bbox, f'images/size/{idx}.png')

    for idx, bbox in enumerate(color_bboxes):
        crop(image, bbox, f'images/color/{idx}.png')

    model_recognitions = get_recognitions('images/model/', ocr_model, ocr_converter, ocr_opt)
    color_recognitions = get_recognitions('images/color/', ocr_model, ocr_converter, ocr_opt)
    size_recognitions = get_recognitions('images/size/', ocr_model, ocr_converter, ocr_opt)

    result = {
        'Model': ' '.join(model_recognitions) if model_recognitions else '',
        'Color': ' '.join(color_recognitions) if color_recognitions else '',
        'Size': ' '.join(size_recognitions) if size_recognitions else '',
    }
    return result


def prepare_sticker_bboxes(sticker_bboxes):
    return [
        {
            'field': index,
            'polygon': Polygon(bbox),
        }
        for index, bbox in enumerate(sticker_bboxes)
    ]


def box_ocr(image_path, sticker_bboxes, craft_net, ocr_model, ocr_converter, ocr_opt):
    text_bboxes = get_ocr_bboxes(image_path, craft_net)
    stickers = prepare_sticker_bboxes(sticker_bboxes)

    box_bboxes = []
    for bbox in text_bboxes:
        field = choose_word_area(bbox, stickers)
        if field is None:
            box_bboxes.append(format_bbox(bbox))

    shutil.rmtree('images/box/')
    os.mkdir('images/box/')

    if box_bboxes:
        sort_bboxes(box_bboxes)

    image = Image.open(image_path)

    for idx, bbox in enumerate(box_bboxes):
        crop(image, bbox, f'images/box/{idx}.png')

    box_recognitions = get_recognitions('images/box/', ocr_model, ocr_converter, ocr_opt)

    return ' '.join(box_recognitions) if box_recognitions else ''
            



