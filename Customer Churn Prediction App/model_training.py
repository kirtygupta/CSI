# model_training.py - Part 1: Data Generation
import pandas as pd
import numpy as np


def generate_data(num_samples=1000):
    np.random.seed(42)

    # Generate synthetic features
    tenure = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.round(np.random.uniform(20, 100, size=num_samples), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.8, 1.2), 2)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples)
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], size=num_samples)

    # Generate target (churn) based on some rules
    churn_prob = (
            0.1 +
            0.3 * (contract == 'Month-to-month') +
            0.2 * (internet_service == 'Fiber optic') +
            0.15 * (online_security == 'No') +
            0.001 * (monthly_charges - 50)
    )
    churn = np.random.binomial(1, np.clip(churn_prob, 0, 0.9))

    # Create DataFrame
    data = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'internet_service': internet_service,
        'online_security': online_security,
        'churn': churn
    })

    return data


# Generate and save data
customer_data = generate_data(1000)
customer_data.to_csv('customer_churn.csv', index=False)

# model_training.py - Part 2: Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# Load data
data = pd.read_csv('customer_churn.csv')

# Preprocessing
X = data.drop('churn', axis=1)
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing pipeline
categorical_features = ['contract', 'internet_service', 'online_security']
numeric_features = ['tenure', 'monthly_charges', 'total_charges']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'churn_model.joblib')

print("Model trained and saved successfully!")