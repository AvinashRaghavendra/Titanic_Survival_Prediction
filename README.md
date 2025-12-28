#  Titanic Survival Prediction

##  Project Overview
This project builds a machine learning system to predict whether a passenger
survived the RMS Titanic disaster using demographic and travel-related features.
The system includes data cleaning, feature engineering, exploratory data analysis,
model training, evaluation, and a user-friendly Streamlit interface for interactive
filtering and prediction.

Optional components include real-time prediction using Kafka and containerized
deployment using Docker.

---

##  Objective
- Predict passenger survival using machine learning
- Analyze survival trends based on age, gender, class, and fare
- Provide an interactive UI for filtering and sorting predictions
- Demonstrate real-time ML inference using Kafka (optional)
- Ensure reproducibility using Docker (optional)

---

##  Dataset
The Titanic dataset is sourced from Kaggle and includes the following features:

- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked
- Survived (target variable)

Raw datasets are stored in:
data/raw/

Processed datasets are generated during preprocessing and training.

---

##  Feature Engineering
The following features are engineered:

- **FamilySize** = SibSp + Parch + 1
- **Title extraction** from passenger names (Mr, Mrs, Miss, Rare)
- Categorical encoding for:
  - Sex
  - Embarked
  - Pclass
  - Title

Unused or high-missing columns (Cabin, Ticket) are dropped.

---

##  Exploratory Data Analysis (EDA)
EDA is performed in:

notebooks/EDA_Titanic.ipynb

Key insights include:
- Females had significantly higher survival rates
- First-class passengers survived more often
- Children had higher survival probability
- Fare and class strongly influenced survival
- Strong interaction between gender and passenger class

---

##  Machine Learning Models
The following models are trained and evaluated:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- 5-Fold Cross-Validation

The best-performing model is saved to:
models/titanic_model.joblib

---

## Streamlit Web Application
The Streamlit app allows users to:

- Filter passengers by:
  - Passenger Class
  - Gender
  - Age Range
- View survival probabilities
- Sort predictions dynamically

### Run Streamlit App
```bash
streamlit run streamlit_app/app.py

App runs at:

http://localhost:8501

Kafka – Real-Time Prediction (Optional)
Kafka is used to simulate real-time passenger data streaming.

Kafka Components
Producer: Streams passenger data row-by-row
Consumer: Applies preprocessing and predicts survival probability

Run Kafka Consumer
python src/kafka/consumer.py

Run Kafka Producer
python src/kafka/producer.py

Kafka topic used:
titanic_passengers

Docker Deployment (Optional)
The application can be containerized using Docker.

Build Docker Image
docker build -t titanic-app -f docker/Dockerfile .

Run Docker Container
docker run -p 8501:8501 titanic-app

Access the app at:
http://localhost:8501


Project Structure
Titanic_Survival_Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── EDA_Titanic.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── kafka/
│       ├── producer.py
│       └── consumer.py
├── models/
│   ├── titanic_model.joblib
│   └── feature_columns.joblib
├── streamlit_app/
│   └── app.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md

Video Demonstration
A 5–10 minute video demonstration includes:

Project overview

EDA walkthrough

Model training & evaluation

Streamlit UI demo

Kafka real-time prediction (optional)

(Video link to be added)

Technologies Used
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Kafka

Docker

Final Notes

This project follows clean coding practices, modular design, and reproducible machine learning workflows, aligned with industry standards and real-world ML deployment scenarios.

---