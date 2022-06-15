import os
import datetime
from dateutil import parser

from export.get_data_from_db import get_applications_from_db, update_application, activate_model
from helpers.rabbitmq_messenger import send_restart_python_api_messages


def cron_upload(command_to_restart_app, command_to_restart_cron_task):
    applications = get_applications_from_db()
    upload_applications = [application for application in applications
                           if 'status' in application and application['status'] == 'trained' and
                           # TODO: Here remove timedelta if today
                           parser.parse(application['date_start']).date() <= datetime.date.today() + datetime.timedelta(days=1) <=
                           parser.parse(application['date_end']).date() and
                           'model' in application]
    print(upload_applications)
    if upload_applications:
        try:
            model_id = upload_applications[0]['model']
            # here we will activate newly trained model to all existing computers in the system
            activate_model(model_id)

            # restart cron task
            os.system(command_to_restart_cron_task)
            # here we restart all python api on each computer
            send_restart_python_api_messages()

        except Exception as e:
            for app in upload_applications:
                update_application(app['_id'], 'failed to upload model')
            print('Failed to upload a model :(')
            print(e)

    else:
        print("No models for today!")

