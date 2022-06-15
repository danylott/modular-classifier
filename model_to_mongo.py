from pymongo import MongoClient
from config.config import MONGO_URL
import datetime

PATH = 'models/model_88_auto.pth'
CLASSES = ['mare', 'arousa', 'armani', 'brandy', 'bryan', 'clarks', 'core', 'courtset', 'dacota', 'dilceida', 'emmshu', 'fluchos', 'fredperry', 'geox', 'gg', 'harmoni', 'heritage', 'krackcore', 'krackcore2', 'krackkids', 'levis_kids', 'lol', 'love', 'newbalance', 'nike', 'nike_sb', 'pepejeans', 'positivity', 'sacut', 'sandrafontan', 'sotavento', 'stonefly', 'unisa', 'vans', 'velilla', 'violeta', 'walk', 'krackcore6', 'angel_infantes', 'bprivate', 'bryan2', 'buonarotti', 'coolway', 'cossimo2', 'dulceida2', 'isteria', 'jenker', 'krack_by_ied', 'krackcore7', 'krack_harmony', 'krack_harmony2', 'krackcore3', 'krackcore4', 'krackcore5', 'krackcore8', 'krackcore9', 'krackkids2', 'krackkids3', 'krackkids4', 'milatrend', 'nice', 'roberto_torreta', 'roberto_toretta2', 'roberto_toretta3', 'rocio_camacho', 'sandra_fontan3', 'sandra_fontan4', 'sandrafontan2', 'viguera', 'wonders', 'sandra_fontan5', 'nice2', 'harmony2', 'krack_heritage', 'krack_heritage2', 'sandra_fontan6', 'krackcore10', 'dulceida3', 'krack_heritage3', 'krack398', 'mou', 'mou2', 'krackcore11', 'krackcore12', 'krackcore13', 'krackcore14', 'roberto_torretta4', 'rocio_camacho2']

client = MongoClient(host=MONGO_URL)
db = client.krack
db.models.insert_one({
    'path': PATH,
    'classes': CLASSES,
    'date_created': str(datetime.datetime.now())
})
print('Successfully added model!')
