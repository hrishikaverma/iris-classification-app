import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

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
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train models and store accuracy
accuracy = {}
model_reports = {}
conf_matrices = {}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, preds)
    model_reports[name] = classification_report(y_test, preds, target_names=iris.target_names, output_dict=True)
    conf_matrices[name] = confusion_matrix(y_test, preds)

    # Feature importance for tree models
    if hasattr(model, "feature_importances_"):
        feature_importances[name] = model.feature_importances_

# Custom CSS for styling
st.markdown("""<style>/* ... (your full CSS remains unchanged) ... */</style>""", unsafe_allow_html=True)

# Title & image
st.markdown('<div class="title">🌸 Iris Flower Classification Web App</div>', unsafe_allow_html=True)
st.markdown('<img class="app-image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg" alt="Iris Flower" width="320">', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<h2>Model & Input Settings</h2>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Choose model for prediction:", list(models.keys()))
selected_model = models[model_choice]
st.sidebar.markdown("### Input Flower Measurements:")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Accuracy chart
st.markdown('<div class="subheader">Model Comparison Accuracy:</div>', unsafe_allow_html=True)
acc_df = pd.DataFrame(list(accuracy.items()), columns=['Model', 'Accuracy'])
chart = alt.Chart(acc_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
    x=alt.X('Model', sort='-y'),
    y=alt.Y('Accuracy', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Model', legend=None),
    tooltip=['Model', alt.Tooltip('Accuracy', format='.2f')]
).properties(width=650, height=320)
st.altair_chart(chart, use_container_width=True)
for name, acc in accuracy.items():
    st.markdown(f'<div class="accuracy-text">- <b>{name}</b>: {acc:.2f}</div>', unsafe_allow_html=True)

# Predict
if st.button("Predict Species"):
    with st.spinner('Predicting...'):
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = selected_model.predict(sample)
        species_name = iris.target_names[prediction[0]]
        st.success(f"Predicted Species: **{species_name}**")

# Expander section
with st.expander("🔍 See Model Details & Reports"):
    st.subheader("🔢 Classification Report")
    report_df = pd.DataFrame(model_reports[model_choice]).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='BuGn'))

    st.subheader("🧮 Confusion Matrix")
    cm = conf_matrices[model_choice]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    if model_choice in feature_importances:
        st.subheader("🌟 Feature Importance")
        imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances[model_choice]
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

with st.expander("📁 Dataset Explorer"):
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.dataframe(df.describe())

    st.write("### Target Labels")
    st.markdown(", ".join([f"**{i}**: {label}" for i, label in enumerate(iris.target_names)]))

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit & scikit-learn</div>', unsafe_allow_html=True)
import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

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
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train models and store accuracy
accuracy = {}
model_reports = {}
conf_matrices = {}
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy[name] = accuracy_score(y_test, preds)
    model_reports[name] = classification_report(y_test, preds, target_names=iris.target_names, output_dict=True)
    conf_matrices[name] = confusion_matrix(y_test, preds)

    # Feature importance for tree models
    if hasattr(model, "feature_importances_"):
        feature_importances[name] = model.feature_importances_

# Custom CSS for styling
st.markdown("""<style>/* ... (your full CSS remains unchanged) ... */</style>""", unsafe_allow_html=True)

# Title & image
st.markdown('<div class="title">🌸 Iris Flower Classification Web App</div>', unsafe_allow_html=True)
st.markdown('<img class="app-image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg" alt="Iris Flower" width="320">', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<h2>Model & Input Settings</h2>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox("Choose model for prediction:", list(models.keys()))
selected_model = models[model_choice]
st.sidebar.markdown("### Input Flower Measurements:")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Accuracy chart
st.markdown('<div class="subheader">Model Comparison Accuracy:</div>', unsafe_allow_html=True)
acc_df = pd.DataFrame(list(accuracy.items()), columns=['Model', 'Accuracy'])
chart = alt.Chart(acc_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
    x=alt.X('Model', sort='-y'),
    y=alt.Y('Accuracy', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Model', legend=None),
    tooltip=['Model', alt.Tooltip('Accuracy', format='.2f')]
).properties(width=650, height=320)
st.altair_chart(chart, use_container_width=True)
for name, acc in accuracy.items():
    st.markdown(f'<div class="accuracy-text">- <b>{name}</b>: {acc:.2f}</div>', unsafe_allow_html=True)

# Predict
if st.button("Predict Species"):
    with st.spinner('Predicting...'):
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = selected_model.predict(sample)
        species_name = iris.target_names[prediction[0]]
        st.success(f"Predicted Species: **{species_name}**")

# Expander section
with st.expander("🔍 See Model Details & Reports"):
    st.subheader("🔢 Classification Report")
    report_df = pd.DataFrame(model_reports[model_choice]).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='BuGn'))

    st.subheader("🧮 Confusion Matrix")
    cm = conf_matrices[model_choice]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    if model_choice in feature_importances:
        st.subheader("🌟 Feature Importance")
        imp_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances[model_choice]
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature'))

with st.expander("📁 Dataset Explorer"):
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.dataframe(df.describe())

    st.write("### Target Labels")
    st.markdown(", ".join([f"**{i}**: {label}" for i, label in enumerate(iris.target_names)]))

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit & scikit-learn</div>', unsafe_allow_html=True)
