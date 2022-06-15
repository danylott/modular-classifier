import argparse
import glob
import os
import cv2
import time
import sys
import random
import requests
import numpy as np
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import config.config as config

classes = config.CLASSES


def get_percent_black_pixels(img):
    # get all non black Pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnt_not_black = cv2.countNonZero(gray)

    # get pixel count of image
    height, width, _ = img.shape
    cnt_pixels = height * width

    # compute all black pixels
    cnt_black = cnt_pixels - cnt_not_black
    return cnt_black / cnt_pixels


def get_best_prediction(det_crop, psp_crop, threshold):
    psp_black_percent = get_percent_black_pixels(psp_crop)
    det_black_percent = get_percent_black_pixels(det_crop)
    print("psp_black_percent", psp_black_percent)
    print("det_black_percent", det_black_percent)
    if psp_black_percent - det_black_percent < threshold:
        return psp_crop
    
    else:
        return det_crop


def get_predictor(args, model_path, num_classes):
    cfg = get_cfg()
    cfg.merge_from_file("base-rcnn.yaml")
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TEST = "no_dataset"
    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if not config.USE_GPU:
        cfg.MODEL.DEVICE = "cpu"
    else: 
        cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    return predictor


offset = 5


def cut_sticker(img):
    (h, w) = img.shape[0:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    return img[y + offset : y + h - offset, x + offset : x + w - offset]


def process_image(predictor, img, label, args, only_bbox=False):
    start_time = time.time()
    print('starting class predictor')
    outputs = predictor(img)
    print('ending class predictor')
    if args.verbose:
        print(f"{label}:", "{:.2f}s".format(time.time() - start_time))

    instances = outputs["instances"]
    if len(instances) > 0:
        res = 0
        if args.filter_classes:
            found = False
            for i in range(len(instances)):
                # print(
                #     classes[instances.pred_classes[i].item()],
                #     instances.scores[i].item(),
                # )
                if classes[instances.pred_classes[i].item()] in args.filter_classes:
                    res = i
                    found = True
                    break
            if not found:
                print("not_found in_filter, but found:", classes[instances.pred_classes[0].item()])
                return {
                    "found": True,
                    "className": classes[instances.pred_classes[0].item()],
                    "message": f"Class {classes[instances.pred_classes[0].item()]} was found, but not in filter list",
                }

        score = instances.scores[res].item()
        pred_class = instances.pred_classes[res].item()
        if config.USE_GPU:
            x1, y1, x2, y2 = instances.pred_boxes[res].tensor[0].cpu().numpy().astype(int)
        else:
            x1, y1, x2, y2 = instances.pred_boxes[res].tensor[0].numpy().astype(int)
        
        bbox = (x1, y1, x2, y2)
        
        if only_bbox:
            return bbox
        # crop = img[y1:y2, x1:x2]
        # cv2.imwrite(f'./labels/{classes[pred_class]}{random.random()}.jpg', crop)
        if config.USE_GPU:
            mask = instances.pred_masks[res].cpu().numpy()
        else:
            mask = instances.pred_masks[res].numpy()
        img_masked = img
        img_masked[~mask, :] = [0, 0, 0]
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
        print(classes[pred_class], score)
        # if args.save_crop:
        #     cv2.imwrite(args.save_crop, crop)
        if args.verbose:
            # cv2.imshow("masked", img_masked)
            cv2.imshow("corrected", crop)
            cv2.waitKey(0)
        return {"found": True, "className": classes[pred_class], "score": score, "crop": crop, "bbox": bbox}
    else:
        print("not_found")
        return {"found": False, "message": "No stickers found"}

    if args.verbose:
        v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get("no_data"),
            instance_mode=ColorMode.IMAGE_BW,
            scale=0.5,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("img", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)


def get_parser():
    parser = argparse.ArgumentParser(description="Krack detectron2 model launcher")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--stream", help="Send stream as stdin using pipe", action="store_true"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-crop", type=str)
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--filter-classes", nargs="+",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    start_time = time.time()
    args = get_parser().parse_args()
    predictor = get_predictor(args)

    if args.verbose:
        print("predictor init:", "{:.2f}s".format(time.time() - start_time))

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in args.input:
            img = cv2.imread(path)
            process_image(predictor, img, path, args)
    elif args.stream:
        camera = cv2.VideoCapture("/dev/stdin")
        while True:
            (grabbed, frame) = camera.read()
            if not grabbed:
                break
            process_image(predictor, frame, "stream", args)
        camera.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        basename = os.path.basename(args.video_input)

        success = True
        frame = 0
        while success:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = video.read()
            if success:
                process_image(image, f"{basename}, frame {frame}", args)
                frame += frames_per_second * 3
