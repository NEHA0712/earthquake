import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🌸 Iris Flower Prediction App")

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

st.write("### Sample Data", df.head())

# Train model
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"✅ Model Accuracy: **{accuracy:.2f}**")

# User input
st.write("### Enter flower measurements:")
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
if st.button("Predict Species"):
    pred = model.predict([[sl, sw, pl, pw]])
    species = iris.target_names[pred[0]]
    st.success(f"🌼 Predicted Iris Species: **{species}**")
