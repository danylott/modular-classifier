import datetime
import time
import os
from argparse import Namespace
from dateutil import parser

from export.get_data_from_db import get_applications_from_db, insert_model, insert_training, update_application
from export.export_to_coco import export_to_coco
from export.split_coco import split_coco

from helpers.train import train_model

import config.config as config


def cron_train(max_iterations: int, command_to_stop_app: str):
    print("Preparing for model training!")
    applications = get_applications_from_db()
    train_applications = [application for application in applications if
                          parser.parse(application['date_start']).date()
                          <= datetime.date.today() + datetime.timedelta(days=1) <= parser.parse(
                              application['date_end']).date()]

    if train_applications:
        # stop flask-app
        try:
            os.system(command_to_stop_app)

        except Exception as e:
            print('Failed to stop flask app :(')
            print(e)
            return

        try:
            target_classes = set()
            for app in train_applications:
                update_application(app['_id'], 'in progress')
                for cls in app['classes']:
                    target_classes.add(cls)

            target_classes = list(target_classes)
            print('Target classes to train:', target_classes)
            # TODO set target application status to 'in progress'
            # export data to coco
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
                                          config.PATH_TO_TEST_JSON, target_classes, model_to_save, max_iterations)

            model_in_db = insert_model(model)
            training_in_db = insert_training(training)
            for app in train_applications:
                update_application(app['_id'], 'trained',
                                   model_in_db.inserted_id, training_in_db.inserted_id)

        except Exception as e:
            for app in train_applications:
                update_application(app['_id'], 'failed to train')
            print('Failed to train applications :(')
            print(e)

    else:
        print('No incoming applications for tomorrow!')


if __name__ == "__main__":
    cron_train(100, "pm2 stop flask-app")
