import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Features & Labels
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier()
}

# Train models and store accuracy
accuracy = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, preds)

# Streamlit UI
st.title("🌸 Iris Flower Classification Web App")

st.write("### Model Comparison Accuracy:")
for name, acc in accuracy.items():
    st.write(f"- **{name}**: {acc:.2f}")

# Select model for prediction
model_choice = st.selectbox("Choose model for prediction:", list(models.keys()))
selected_model = models[model_choice]

# Input sliders for features
st.write("### Input Flower Measurements:")

sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Predict button
if st.button("Predict Species"):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = selected_model.predict(sample)
    species_name = iris.target_names[prediction[0]]
    st.success(f"Predicted Species: **{species_name}**")
