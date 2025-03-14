import json
import time
from kafka import KafkaConsumer, KafkaProducer
from flask import Flask, jsonify, request
from threading import Thread
from queue import Queue
import random

app = Flask(__name__)

# Configuration for Kafka
KAFKA_BROKER = 'localhost:9092'
INPUT_TOPIC = 'input_topic'
OUTPUT_TOPIC = 'output_topic'

# Queue for holding processed data
data_queue = Queue()

def produce_data():
    producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)
    while True:
        # Simulate data
        data = {'value': random.randint(1, 100)}
        producer.send(INPUT_TOPIC, json.dumps(data).encode('utf-8'))
        print(f"Produced: {data}")
        time.sleep(1)

def consume_data():
    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        group_id='data_processor',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        print(f"Consumed: {message.value}")
        processed_data = process_data(message.value)
        data_queue.put(processed_data)
        send_result_to_kafka(processed_data)

def process_data(data):
    # Simulate some processing
    processed_value = data['value'] * 2
    return {'original': data['value'], 'processed': processed_value}

def send_result_to_kafka(data):
    producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)
    producer.send(OUTPUT_TOPIC, json.dumps(data).encode('utf-8'))
    print(f"Sent to Kafka: {data}")

@app.route('/results', methods=['GET'])
def get_results():
    results = []
    while not data_queue.empty():
        results.append(data_queue.get())
    return jsonify(results)

def start_flask():
    app.run(port=5000)

if __name__ == "__main__":
    # Start the data producer and consumer in separate threads
    Thread(target=produce_data).start()
    Thread(target=consume_data).start()
    start_flask()