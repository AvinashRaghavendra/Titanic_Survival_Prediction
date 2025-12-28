import json
import time
import pandas as pd
from kafka import KafkaProducer

# -----------------------------
# Kafka Configuration
# -----------------------------
KAFKA_TOPIC = "titanic_passengers"
KAFKA_SERVER = "localhost:9092"

# -----------------------------
# Create Kafka Producer
# -----------------------------
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# -----------------------------
# Load Titanic Data
# -----------------------------
df = pd.read_csv("../../data/raw/train.csv")

# Drop target column for streaming
df = df.drop(columns=["Survived"])

print("🚀 Starting Kafka Producer...")
print(f"Sending data to topic: {KAFKA_TOPIC}")

# -----------------------------
# Stream Data Row by Row
# -----------------------------
for _, row in df.iterrows():
    message = row.to_dict()
    producer.send(KAFKA_TOPIC, value=message)
    print(f"Sent PassengerId: {message['PassengerId']}")
    time.sleep(2)  # simulate real-time streaming

producer.flush()
producer.close()
