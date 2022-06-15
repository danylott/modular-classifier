import os
import detectron2
from detectron2.utils.logger import setup_logger
import datetime
import pika
setup_logger()

import numpy as np
import cv2
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog

from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances

from pymongo import MongoClient

import config.config as config

def log(body):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='train_log', durable=False)
    channel.basic_publish(
        exchange='',
        routing_key='train_log',
        body=body
    )
    print(body)
    connection.close()

def train_model(data_dir, train_path, test_path, classes, path_to_save_model, max_iteraions=1200):
    log('Training started.')
    
    log(test_path)
    log(str(classes))

    DatasetCatalog.clear()
    register_coco_instances("krack_train", {}, train_path, data_dir)
    register_coco_instances("krack_test", {}, test_path, data_dir)
    metadata = MetadataCatalog.get("krack_train")
    dataset_dicts = DatasetCatalog.get("krack_train")

    log(str(metadata.thing_classes))

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (["krack_train"])
    cfg.DATASETS.TEST = (["krack_test"])
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "/home/su/work/krack/krack-python/models/model_final_f10217.pkl"    # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = max_iteraions
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.DEVICE = 'cpu'

    start = str(datetime.datetime.now())

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    path_to_model = cfg.OUTPUT_DIR + '/model_final.pth'
    training = {
        # 'detectron_config' : str(cfg),
        'MAX_ITER': cfg.SOLVER.MAX_ITER,
        # 'NUM_WORKERS': cfg.DATALOADER.NUM_WORKERS,
        'TRAIN_TEST_SPLIT_COEF': config.TRAIN_TEST_SPLIT_COEF,
        'TRAIN_TEST_SPLIT_MIN_IMAGES': config.TRAIN_TEST_SPLIT_MIN_IMAGES,
        'classes': classes,
        'path_to_model': path_to_save_model,
        'date_started': start,
        'date_ended': str(datetime.datetime.now()),
    }

    model = {
        'path': path_to_save_model,
        'classes': classes,
        'date_created': str(datetime.datetime.now()),
    }

    os.rename(path_to_model, path_to_save_model)
    log('successfully trained! Model:{}'.format(model))
    return training, model
