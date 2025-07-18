#  streamlit run app.py
# pip install -r requirements.txt


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Constants
CATEGORICAL_FEATURES = ['contract', 'internet_service', 'online_security']
NUMERIC_FEATURES = ['tenure', 'monthly_charges', 'total_charges']

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")


# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')


@st.cache_data
def load_data():
    return pd.read_csv('customer_churn.csv')


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Predict Churn", "Model Analysis"])

# Load resources
model = load_model()
data = load_data()

# Home page
if page == "Home":
    st.title("Customer Churn Prediction App")
    st.image("https://cdn-icons-png.flaticon.com/512/2742/2742169.png", width=150)
    st.write("""
    This application helps predict customer churn using machine learning.

    **Features:**
    - Explore the customer dataset
    - Predict churn for individual customers
    - Analyze model performance and metrics
    """)

    st.write("### Sample Data")
    st.write(data.head())

# Data Explorer page
elif page == "Data Explorer":
    st.title("Data Explorer")

    st.write("### Dataset Overview")
    st.write(f"Number of customers: {len(data)}")
    st.write(f"Churn rate: {data['churn'].mean():.2%}")

    # Interactive filters
    st.sidebar.header("Filters")
    tenure_range = st.sidebar.slider(
        "Tenure (months)",
        min_value=int(data['tenure'].min()),
        max_value=int(data['tenure'].max()),
        value=(0, 72)
    )

    contract_filter = st.sidebar.multiselect(
        "Contract Type",
        options=data['contract'].unique(),
        default=data['contract'].unique()
    )

    # Apply filters
    filtered_data = data[
        (data['tenure'] >= tenure_range[0]) &
        (data['tenure'] <= tenure_range[1]) &
        (data['contract'].isin(contract_filter))
        ]

    st.write(f"Showing {len(filtered_data)} customers")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Churn by Contract Type")
        contract_churn = filtered_data.groupby('contract')['churn'].mean().reset_index()
        st.bar_chart(contract_churn.set_index('contract'))

    with col2:
        st.write("#### Monthly Charges Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_data['monthly_charges'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Filtered Data")
    st.write(filtered_data)

# Prediction page
elif page == "Predict Churn":
    st.title("Churn Prediction")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (months)", min_value=1, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                            value=tenure * monthly_charges)

        with col2:
            contract = st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year'])
            internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", options=['Yes', 'No', 'No internet service'])

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame(
            [[tenure, monthly_charges, total_charges, contract, internet_service, online_security]],
            columns=data.columns[:-1])

        # Make prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        # Display results
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 High churn risk ({proba[1]:.1%} probability)")
        else:
            st.success(f"✅ Low churn risk ({proba[0]:.1%} probability)")

        # Show probabilities
        st.write("### Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Class': ['Stay', 'Churn'],
            'Probability': proba
        })
        st.bar_chart(prob_df.set_index('Class'))

# Model Analysis page
elif page == "Model Analysis":
    st.title("Model Analysis")

    # Load test data (or split again)
    X = data.drop('churn', axis=1)
    y = data['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Stay', 'Predicted Churn'],
                yticklabels=['Actual Stay', 'Actual Churn'],
                ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Feature Importance (if available)
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        st.write("### Feature Importance")

        # Get feature names after one-hot encoding
        preprocessor = model.named_steps['preprocessor']
        ohe = preprocessor.named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
        all_features = NUMERIC_FEATURES + list(cat_features)

        importance = model.named_steps['classifier'].feature_importances_
        feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importance})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax)
        ax.set_title('Feature Importance')
        st.pyplot(fig)