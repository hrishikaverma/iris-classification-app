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
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
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
    /* Global font and background */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(120deg, #e0f7fa, #80deea);
        color: #013440;
    }
    /* Title styling */
    .title {
        color: #00796b;
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0;
        padding-top: 10px;
        font-family: 'Segoe UI Black', sans-serif;
        text-shadow: 1px 1px 2px #004d40;
    }
    /* Subtitle */
    .subheader {
        color: #004d40;
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
        text-align: center;
    }
    /* Accuracy text */
    .accuracy-text {
        color: #004d40;
        font-size: 18px;
        margin: 5px 0;
    }
    /* Footer */
    .footer {
        font-size: 13px;
        color: #004d40aa;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 10px;
        font-style: italic;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #00695c;
        color: #e0f2f1;
        padding: 20px;
        border-radius: 10px;
        font-weight: 600;
    }
    /* Sidebar header */
    .sidebar .sidebar-content h2 {
        color: #a7ffeb;
        font-weight: 900;
        font-size: 24px;
    }
    /* Sliders styling */
    .stSlider > div {
        color: #004d40;
    }
    /* Button styling */
    div.stButton > button:first-child {
        background-color: #00796b;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 8px 20px;
        transition: background-color 0.3s ease;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #004d40;
        cursor: pointer;
    }
    /* Image styling */
    .app-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 15px;
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }
    /* Expander customization */
    details > summary {
        font-weight: 700;
        font-size: 18px;
        color: #00796b;
        cursor: pointer;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    details[open] > summary {
        color: #004d40;
    }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown('<div class="title">🌸 Iris Flower Classification Web App</div>', unsafe_allow_html=True)

# Add image with styling
st.markdown(
    '<img class="app-image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg" alt="Iris Flower" width="320">',
    unsafe_allow_html=True,
)

# Sidebar content
st.sidebar.markdown('<h2>Model & Input Settings</h2>', unsafe_allow_html=True)

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

# Bar chart using Altair with smooth corners and tooltip
chart = alt.Chart(acc_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
    x=alt.X('Model', sort='-y'),
    y=alt.Y('Accuracy', scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Model', legend=None),
    tooltip=['Model', alt.Tooltip('Accuracy', format='.2f')]
).properties(width=650, height=320)

st.altair_chart(chart, use_container_width=True)

# Also show accuracy as styled text list
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
