import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
import joblib
import os
import plotly.express as px

# Ensure to use appropriate version and recheck dependencies
@st.cache_data
def load_data():
    data = pd.read_csv('app/dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    Target = 'Diabetes_binary'
    Selected_features = ['GenHlth' ,'CholCheck' ,'HighBP' ,'AnyHealthcare' ,'PhysActivity' ,
                         'BMI' ,'HighChol' ,'Age' ,'Fruits' ,'Income' ,'DiffWalk' ,
                         'Education' ,'HeartDiseaseorAttack','PhysHlth' , 'Smoker']
    X = data[Selected_features].astype(int)
    y = data[Target].astype(int)
    return X , y

@st.cache_resource
def train_model(X, y):
    cat = CatBoostClassifier(verbose=0, allow_writing_files=False)
    lgb = LGBMClassifier(verbose=-1)
    xgb = XGBClassifier()

    model = [('cat', cat), ('lgb', lgb), ('xgb', xgb)]

    clf = VotingClassifier(model, voting='soft', n_jobs=-1)

    clf.fit(X, y)

    # Ensure the model directory exists
    model_dir = 'app/model'
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    joblib.dump(clf, os.path.join(model_dir, 'model.pkl'))

    return clf

def load_or_train_model(X, y):
    model_path = 'app/model/model.pkl'
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        st.session_state.model = clf
    elif not st.session_state.get('model'):
        st.session_state.model = train_model(X, y)
    return st.session_state.model

def get_user_input():
    with st.sidebar.expander('Input Feature Values', expanded=False):
        HighBP = st.selectbox("HighBP: Have you been told you have high blood pressure?", [0, 1], help="0: No, 1: Yes")
        HighChol = st.selectbox("HighChol: Have you been told your cholesterol is high?", [0, 1], help="0: No, 1: Yes")
        CholCheck = st.selectbox("CholCheck: Cholesterol check within past five years?", [0, 1], help="0: No, 1: Yes")
        BMI = st.slider("BMI: Body Mass Index", 10.0, 50.0, 25.0)
        Smoker = st.selectbox("Smoker: Have you smoked at least 100 cigarettes in your life?", [0, 1], help="0: No, 1: Yes")
        HeartDiseaseorAttack = st.selectbox("HeartDiseaseorAttack: Have you had coronary heart disease or myocardial infarction?", [0, 1], help="0: No, 1: Yes")
        PhysActivity = st.selectbox("PhysActivity: Do you engage in physical activity or exercise?", [0, 1], help="0: No, 1: Yes")
        Fruits = st.selectbox("Fruits: Do you consume fruit 1 or more times per day?", [0, 1], help="0: No, 1: Yes")
        Income = st.selectbox("Income: Annual household income level", [1, 2, 3, 4, 5, 6, 7, 8], help="1: <$10,000, 8: >=$75,000")
        DiffWalk = st.selectbox("DiffWalk: Do you have serious difficulty walking or climbing stairs?", [0, 1], help="0: No, 1: Yes")
        Education = st.selectbox("Education: Highest grade or year of school completed", [1, 2, 3, 4, 5, 6], help="1: Never attended school, 6: College graduate")
        GenHlth = st.selectbox("GenHlth: General health rating", [1, 2, 3, 4, 5], help="1: Excellent, 5: Poor")
        PhysHlth = st.slider("PhysHlth: Number of days physical health was not good", 0, 30, 0)
        Age = st.selectbox("Age: Age category", list(range(1, 15)), help="1: 18-24 years, 14: 80+ years")
        AnyHealthcare = st.selectbox("AnyHealthcare: Do you have any health care coverage?", [0, 1], help="0: No, 1: Yes")

        user_input = {
            'GenHlth': GenHlth,
            'CholCheck': CholCheck,
            'HighBP': HighBP,
            'AnyHealthcare': AnyHealthcare,
            'PhysActivity': PhysActivity,
            'BMI': BMI,
            'HighChol': HighChol,
            'Age': Age,
            'Fruits': Fruits,
            'Income': Income,
            'DiffWalk': DiffWalk,
            'Education': Education,
            'HeartDiseaseorAttack': HeartDiseaseorAttack,
            'PhysHlth': PhysHlth,
            'Smoker': Smoker
        }

    return pd.DataFrame([user_input]).astype(int)

def plot_graphs(X, y, model):
    # Predict probabilities for the test set
    y_probs = model.predict_proba(X)
    y_probs_positive_class = y_probs[:, 1]

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, y_probs_positive_class)
    auprc = auc(recall, precision)
    
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker='.')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (AUC = {auprc:.2f})')
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_probs_positive_class)
    roc_auc = roc_auc_score(y, y_probs_positive_class)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='.')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve (AUC = {roc_auc:.2f})')
    st.pyplot(fig)

    # Confusion Matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def predict_utils(X , y , clf , input_data):
    if st.sidebar.button('Predict'):
        y_pred = clf.predict(input_data)
        y_probs = clf.predict_proba(input_data)
        y_probs_df = pd.DataFrame(y_probs, columns=["No Diabetes Probability", "Diabetes Probability"])
        with st.expander("Result" , expanded= True):
            st.write(y_probs_df)
            prediction = "Diabetes" if y_pred[0] == 1 else "No Diabetes"

            if prediction == "Diabetes":
                st.error("The model predicts: **Diabetes**")
                st.snow()
            else:
                st.success("The model predicts: **No Diabetes**")
                st.balloons()

def app():
    X, y = load_data()
    clf = load_or_train_model(X, y)
    input_data = get_user_input()
    predict_utils(X , y , clf , input_data)

    with st.expander('Introduction', expanded=True):
        st.header('Introduction')
        st.image('app/dataset/photo.jpg', caption='Diabetes Prediction')
        st.markdown('''Diabetes prediction uses machine learning to identify
                     individuals at risk before symptoms develop. By analyzing 
                    factors like glucose levels, BMI, and family history, predictive 
                    models help in early diagnosis and targeted prevention. 
                    Advanced algorithms improve accuracy and support better management
                     of this widespread condition.''')

    with st.expander('Graphs', expanded=False):
        st.header('Graphs')
        plot_graphs(X, y, clf)

st.markdown("""
    <style>
    .title {
        background: linear-gradient(to bottom, #FF6347, #1E90FF);
        border-radius: 25px;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    </style>
    <h1 class="title">Diabetes Prediction App</h1>
    """, unsafe_allow_html=True)
st.write('')
st.sidebar.header(":blue[Prediction Zone]")
st.sidebar.markdown("Use the inputs below to predict the likelihood of diabetes.")

app()

st.sidebar.info("Adjust the parameters using the sliders and dropdowns in the sidebar.")
