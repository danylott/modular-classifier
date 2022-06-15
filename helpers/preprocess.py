import cv2
import numpy as np
import sys
import os.path
import time
import pytesseract
import random
from PIL import Image


def preprocess(input_file, reader):
    start_time = time.time()
    # Perform OCR
    # without detail = 0 - returns bbox for text on image
    result = reader.readtext(input_file, detail=0)
    print("--- %s seconds for text recognition ---" % (time.time() - start_time))
    
    return '\n'.join(result)
