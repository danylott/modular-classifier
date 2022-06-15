import schedule
import time
import datetime

from cron.cron_train import cron_train
from cron.cron_upload import cron_upload

import config.config as config

max_iterations = config.MAX_TRAIN_ITERATIONS
# max_iterations = 6000

schedule.every().day.at(config.TIME_TO_START_MODEL_TRAINING).do(cron_train, max_iterations, config.COMMAND_TO_STOP_APP)
schedule.every().day.at(config.TIME_TO_START_MODEL_UPLOADING).do(cron_upload, config.COMMAND_TO_RESTART_APP,
                                                                 config.COMMAND_TO_RESTART_CRON_TASK,
                                                                 config.COMPUTER_POSITION_LIST)

while True:
    # NOTICE, that this cron_task will be running only on the server machine
    print(f'check: {datetime.datetime.now()}')
    schedule.run_pending()
    time.sleep(60)  # wait one minute
