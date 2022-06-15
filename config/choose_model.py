from pymongo import MongoClient
from config import constants
from bson import ObjectId


def choose_model():
    client = MongoClient(host=constants.MONGO_URL)
    db = client.krack
    if 'models' not in db.collection_names():
        return False

    if 'computers' not in db.collection_names():
        return False

    computer = db.computers.find_one({"position": constants.COMPUTER_POSITION})
    if computer is None:
        return False

    model = db.models.find_one({"_id": ObjectId(computer['active_model'])})
    if model is None:
        return False

    return model['classes'], model['path']
