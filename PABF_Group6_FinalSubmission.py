import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.quantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

class TargetEncoder1D(BaseEstimator, TransformerMixin):
    def __init__(self, min_samples_leaf=20, smoothing=10):
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.Series(X.ravel())
        y = pd.Series(y)
        self.global_mean_ = y.mean()

        stats = y.groupby(X).agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(stats["count"] - self.min_samples_leaf) / self.smoothing))
        enc = self.global_mean_ * (1 - smoothing) + stats["mean"] * smoothing
        self.mapping_ = enc.to_dict()
        return self

    def transform(self, X):
        X = pd.Series(X.ravel())
        return X.map(self.mapping_).fillna(self.global_mean_).values.reshape(-1, 1)

import joblib
model = joblib.load("churn_model.pkl")


st.title("📊 Telecom Churn Prediction App")
st.write("""
Upload a customer dataset (CSV format) and the model will generate 
**churn probability scores** for each customer.
""")

uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded is not None:
    # Read uploaded file
    data = pd.read_csv(uploaded)
    
    # IMPORTANT FIX: Make column names lowercase
    data.columns = data.columns.str.lower()
    
    st.subheader("Preview of Uploaded Data")
    st.write(data.head())

    # Predict probabilities
    churn_prob = model.predict_proba(data)[:, 1]
    data["churn_probability"] = churn_prob

    st.subheader("Predicted Churn Probabilities")
    st.write(data.head())

    # Download results
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )