import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Dummy dataset
data = pd.DataFrame({
    "months_paid": [1, 2, 3, 4],
    "churn": [1, 1, 0, 0]
})

X = data[["months_paid"]]
y = data["churn"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "ml/models/churn_v1.joblib")
print("Model trained and saved to ml/models/churn_v1.joblib")
