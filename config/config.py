from config.choose_model import choose_model
from config.constants import *


model = choose_model()
if not model:
    NUM_CLASSES = NUM_CLASSES_DEFAULT
    MODEL_WEIGHT_PATH = MODEL_WEIGHT_PATH_PREFIX + MODEL_WEIGHT_PATH_DEFAULT
    CLASSES = CLASSES_DEFAULT
else:
    NUM_CLASSES = len(model[0])
    MODEL_WEIGHT_PATH = MODEL_WEIGHT_PATH_PREFIX + model[1]
    CLASSES = model[0]
