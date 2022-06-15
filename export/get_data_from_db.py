from pymongo import MongoClient
from bson import ObjectId

from config.config import MONGO_URL, COMPUTER_POSITION_LIST


client = MongoClient(host=MONGO_URL)
db = client.krack


def get_data_from_db(input_classes):
    classes_db = db.classes.find()
    images_db = db.images.find()

    class_list = []
    image_list = []

    for cls in classes_db:
        if str(cls['name']) in input_classes:
            class_list.append(cls)

    for image in images_db:
        if str(image['cls']) in [str(cls['_id']) for cls in class_list] and 'annotation' in image:
            image_list.append(image)

    return class_list, image_list


def get_applications_from_db():
    return list(db.applications.find())


def insert_model(model):
    return db.models.insert_one(model)


def insert_training(training):
    return db.trainings.insert_one(training)


def update_application(application_id, status, model_id=None, training_id=None):
    db.applications.update_one({
        '_id': application_id
    }, {
        '$set': {
            'status': status,
        }
    }, upsert=False)

    if model_id:
        db.applications.update_one({
            '_id': application_id
        }, {
            '$set': {
                'model': model_id,
            }
        }, upsert=False)

    if training_id:
        db.applications.update_one({
            '_id': application_id
        }, {
            '$set': {
                'training': training_id,
            }
        }, upsert=False)


def activate_model(model_id):
    for computer_position in COMPUTER_POSITION_LIST:
        db.computers.update_one({
            'position': computer_position,
        }, {
            '$set': {
                'active_model': ObjectId(model_id),
            }
        })
        print(f"Activate model for computer_{computer_position}")
