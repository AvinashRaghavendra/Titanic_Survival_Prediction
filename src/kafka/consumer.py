import json
import os
import sys
import pandas as pd
from kafka import KafkaConsumer
from joblib import load

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.data_preprocessing import preprocess

# --------------------------------------------------
# Kafka Configuration
# --------------------------------------------------
KAFKA_TOPIC = "titanic_passengers"
KAFKA_SERVER = "localhost:9092"

# --------------------------------------------------
# Load Model & Feature Schema
# --------------------------------------------------
model = load(os.path.join(PROJECT_ROOT, "models", "titanic_model.joblib"))
feature_columns = load(os.path.join(PROJECT_ROOT, "models", "feature_columns.joblib"))

# --------------------------------------------------
# Create Kafka Consumer
# --------------------------------------------------
consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_SERVER,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print("📡 Kafka Consumer started...")
print(f"Listening to topic: {KAFKA_TOPIC}")

# --------------------------------------------------
# Consume & Predict
# --------------------------------------------------
for message in consumer:
    passenger_data = message.value

    # Convert message to DataFrame
    df = pd.DataFrame([passenger_data])

    # Preprocess incoming data
    X_processed, _ = preprocess(
        df,
        is_train=False,
        feature_columns=feature_columns
    )

    # Remove PassengerId before prediction
    X_model = X_processed.drop(columns=["PassengerId"])

    # Predict survival probability
    survival_prob = model.predict_proba(X_model)[0][1]

    print(
        f"PassengerId {passenger_data['PassengerId']} "
        f"→ Survival Probability: {survival_prob:.4f}"
    )
