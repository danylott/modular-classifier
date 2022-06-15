# Modular OCR classifier `flask-app`

## API

Flask API `api.py`

## Sell script
`predict.py`

Sample usage:
`python3 predict.py --filter-classes nike core pepejeans --input /43classes/IMG_0562_r.JPG`

## Dependencies (Installation):
PTH model file
python3+
- requires python3.7, python3.7-dev packages to run
- `python3 -m venv venv`
- `source venv/bin/activate`
- CPU: `pip install -r requirements-cpu.txt`
- GPU: `pip install -r requirements-gpu.txt`
- Create constants.py from constants.py.sample
- Download weights for PSPNet, Detectron2, CRAFT and OCR
- (optional) populate models collection with active model to use

## Config:
- Everything is stored in constants.py
- USE_GPU/USE_PSPNET
- MODEL_PATHS
- CLASSES

## Script important parts:
- helpers/predict.py: detectron2 image class predictor
- helpers/pspnet_crop.py: PSPNet image cropper
- helpers/ocr.py: OCR performer using craft/ and ocr/

## Cron job:
- Important part of automized model training - trains models 
from applications and activate them
- Run in different process but with the same venv: `python -u cron_task.py`
- Must be active for ever
- Set actual period in cron_task.py
- Send RabbitMQ messages to restart_python_api, when training complete

## Useful links:
- detectron2: https://github.com/facebookresearch/detectron2
- CRAFT OCR: https://github.com/clovaai/CRAFT-pytorch
- Deep OCR: https://github.com/clovaai/deep-text-recognition-benchmark
