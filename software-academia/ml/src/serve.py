import joblib

model = joblib.load("ml/models/churn_v1.joblib")

def predict(months_paid: int):
    return model.predict([[months_paid]])[0]

if __name__ == "__main__":
    print("Prediction for 2 months paid:", predict(2))
