import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.pdfgen import canvas

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

    # PDF Export
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 750, "🌸 Iris Flower Classification Report")
    p.drawString(100, 720, f"Model Used: {model_choice}")
    p.drawString(100, 700, f"Predicted Species: {species_name}")
    p.drawString(100, 680, f"Input Features:")
    p.drawString(120, 660, f"Sepal Length: {sepal_length}")
    p.drawString(120, 640, f"Sepal Width: {sepal_width}")
    p.drawString(120, 620, f"Petal Length: {petal_length}")
    p.drawString(120, 600, f"Petal Width: {petal_width}")
    p.showPage()
    p.save()

    buffer.seek(0)
    st.download_button(label="📄 Download Prediction Report as PDF", data=buffer, file_name="iris_prediction_report.pdf")

# Optional: Dataset Preview
with st.expander("📊 Show Dataset Summary"):
    st.write(df.head())
    st.write(df.describe())

# Optional: Confusion Matrix & Classification Report
if st.checkbox("🔍 Show Confusion Matrix & Classification Report"):
    y_pred = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    st.pyplot(fig)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

# Optional: Feature Importance
if model_choice in ["Random Forest", "Decision Tree"]:
    st.subheader("📈 Feature Importance:")
    importance = selected_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_imp_df.set_index("Feature"))
