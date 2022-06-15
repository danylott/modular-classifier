import pika
import json
from config.constants import COMPUTER_POSITION_LIST


def send_restart_python_api_messages():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    for computer_position in COMPUTER_POSITION_LIST:
        queue_name = f'computer_{computer_position}'
        message_body = {
            "topic": "restart_python_api",
            "payload": {
                "positionId": computer_position,
            }
        }
        channel.queue_declare(queue=queue_name, durable=True)
        channel.basic_publish(exchange='',
                              routing_key=queue_name,
                              body=json.dumps(message_body))
        print(f"Message sent to {queue_name}")

    channel.close()
