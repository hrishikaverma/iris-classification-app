import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    .title {
        color: #4B8BBE;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #306998;
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
    }
    .accuracy-text {
        color: #444444;
        font-size: 18px;
        margin: 5px 0;
    }
    .footer {
        font-size: 12px;
        color: gray;
        text-align: center;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown('<div class="title">🌸 Iris Flower Classification Web App</div>', unsafe_allow_html=True)

# Add image
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg", width=300)

# Sidebar for model selection and inputs
st.sidebar.header("Model & Input Settings")

model_choice = st.sidebar.selectbox("Choose model for prediction:", list(models.keys()))
selected_model = models[model_choice]

st.sidebar.markdown("### Input Flower Measurements:")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Show model accuracies with a chart and text
st.markdown('<div class="subheader">Model Comparison Accuracy:</div>', unsafe_allow_html=True)

acc_df = pd.DataFrame(list(accuracy.items()), columns=['Model', 'Accuracy'])

# Bar chart using Altair
chart = alt.Chart(acc_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
    x=alt.X('Model', sort='-y'),
    y='Accuracy',
    color='Model',
    tooltip=['Model', alt.Tooltip('Accuracy', format='.2f')]
).properties(width=600, height=300)

st.altair_chart(chart)

# Also show accuracy as text list
for name, acc in accuracy.items():
    st.markdown(f'<div class="accuracy-text">- <b>{name}</b>: {acc:.2f}</div>', unsafe_allow_html=True)

# Prediction button and output
if st.button("Predict Species"):
    with st.spinner('Predicting...'):
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = selected_model.predict(sample)
        species_name = iris.target_names[prediction[0]]
        st.success(f"Predicted Species: **{species_name}**")

# Expander with model details
with st.expander("See Model Details"):
    for name, model in models.items():
        st.write(f"**{name}** model details:")
        st.write(model)

# Footer
st.markdown('<div class="footer">Made with ❤️ using Streamlit & scikit-learn</div>', unsafe_allow_html=True)
