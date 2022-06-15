import cv2
from argparse import Namespace
from flask import Flask, request, jsonify

from export.export_to_coco import export_to_coco
from export.get_data_from_db import get_applications_from_db, insert_model, insert_training, update_application

import time
from craft.test import get_craft_net
from ocr.demo import get_ocr_model_converter_opt
from helpers.predict import process_image, get_predictor, get_best_prediction

from helpers.ocr import ocr, box_ocr
from helpers.splitclasses import split_classes
from export.split_coco import split_coco

import config.config as config
from manual.train import train_model

app = Flask(__name__)


@app.route("/", methods=["POST"])
def predict():
    args = Namespace(verbose=False, **request.json)

    img = cv2.imread(args.input)
    response = process_image(main_predictor, img, "api_img", args)
    print("predictor response", response)
    if 'crop' in response:
        if config.USE_BOX_TEXT_OCR and response['score'] < config.SCORE_THRESHOLD_FOR_BOX_OCR:
            main_sticker_bbox = response['bbox']
            code_128_sticker_bbox = process_image(code_128_predictor, img, "code_128_img", args, only_bbox=True)
            return jsonify({"found": False, "message": f"found: {main_sticker_bbox}, {code_128_sticker_bbox}"})
            print("main_sticker_bbox", main_sticker_bbox)
            text = box_ocr(args.input, [main_sticker_bbox, code_128_sticker_bbox], craft_net, ocr_model, ocr_converter, ocr_opt)
            response = {"found": False, "message": "Sticker was found but with bad score, found text on box:\n" + text}
        else:
            det_crop = response['crop']
            if not config.USE_PSPNET:
                cv2.imwrite(args.save_crop, det_crop)

            else:
                from helpers.pspnet_crop import pspnet_crop
                psp_crop = None
                try:
                    psp_crop = pspnet_crop(cropper, args.input, args.save_crop)
                except TypeError as e:
                    print(f"There was an error while pspnet cropping: {e}")

                if psp_crop is None:
                    cv2.imwrite(args.save_crop, det_crop)

                else:
                    result_crop = get_best_prediction(det_crop, psp_crop, config.BLACK_PIXELS_DIFFERENCE)
                    cv2.imwrite(args.save_crop, result_crop)
        
        del response['crop']
        del response['bbox']
        
    return jsonify(response)

 
@app.route("/recognize-text/", methods=["POST"])
def recognize():
    args = Namespace(**request.json)
    data = ocr(args.input, args.markup, craft_net, ocr_model, ocr_converter, ocr_opt)

    print('OCR result: ', data)
    return jsonify(data)


@app.route("/label/", methods=["POST"])
def label():
    from helpers.label_image import label_image
    data = {}
    args = Namespace(**request.json)

    data['annotation'], data['width'], data['height'] = label_image(args.input, args.output, args.annotation)
    # print('labeling', data)
    return jsonify(data)


@app.route("/markup/", methods=["POST"])
def markup():
    from helpers.class_markup import class_markup
    data = {}
    args = Namespace(**request.json)

    data['success'] = class_markup(args.input, args.output, args.markup)
    return jsonify(data)


@app.route("/crop/", methods=["POST"])
def crop():
    from helpers.crop_sticker import crop_sticker
    data = {}
    args = Namespace(**request.json)
    data['output'] = crop_sticker(args.input, args.output, args.annotation)
    return jsonify(data)

@app.route("/train/", methods=["POST"])
def train():
    args = Namespace(**request.json)
    target_classes = args.classes['classes']
    data = {}
    data['status'] = 'ok'
    path_to_coco = config.PATH_TO_COCO_DATASET_FOLDER + f'coco{len(target_classes)}{time.time()}.json'
    export_to_coco(target_classes, path_to_coco)
    # split coco
    split_args = {
        "annotations": path_to_coco,
        "train": config.PATH_TO_TRAIN_JSON,
        "test": config.PATH_TO_TEST_JSON,
        "dataset": config.PATH_TO_IMAGE_DATASET_FOLDER,
        "mintest": config.TRAIN_TEST_SPLIT_MIN_IMAGES,
        "split": config.TRAIN_TEST_SPLIT_COEF,
    }
    args = Namespace(verbose=False, **split_args)
    split_coco(args)

    model_to_save = f'models/model_{len(target_classes)}_{int(time.time())}.pth'

    # train
    training, model = train_model(config.PATH_TO_IMAGE_DATASET_FOLDER, config.PATH_TO_TRAIN_JSON,
                                    config.PATH_TO_TEST_JSON, target_classes, model_to_save, config.MAX_TRAIN_ITERATIONS)

    model_in_db = insert_model(model)
    training_in_db = insert_training(training)
    # for app in train_applications:
    #     update_application(app['_id'], 'trained',
    #                         model_in_db.inserted_id, training_in_db.inserted_id)

    return jsonify(data)

if __name__ == "__main__":
    main_predictor = get_predictor(Namespace(opts="", threshold=0.1), config.MODEL_WEIGHT_PATH, config.NUM_CLASSES)
    print("main predictor initialized")
    if config.USE_BOX_TEXT_OCR:
        code_128_predictor = get_predictor(Namespace(opts="", threshold=0.1), config.MODEL_CODE_128_WEIGHT_PATH, 1)
        print("code 128 predictor initialized")

    craft_net = get_craft_net(config.CRAFT_MODEL_WEIGHT_PATH, config.USE_GPU)
    print("craft net initialized")
    ocr_model, ocr_converter, ocr_opt = get_ocr_model_converter_opt(config.OCR_MODEL_WEIGHT_PATH)
    print("ocr model initialized")
    if config.USE_PSPNET:
        from helpers.pspnet_crop import get_cropper
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        cropper = get_cropper(config.PSPNET_RESNET_MODEL_WEIGHT_PATH)
        print("pspnet cropper initialized")

    app.run()

