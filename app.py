import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🌸 Iris Flower Prediction App")

# Load dataset
df = pd.read_csv("Iris.csv")

st.write("### Sample Data", df.head())

# Train model
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]
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
    st.success(f"🌼 Predicted Iris Species: **{pred[0]}**")
