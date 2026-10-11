import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo
import joblib

print("Fetching UCI Heart Disease dataset...")
# Fetch dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Convert multiclass (0-4) to binary (0 = no disease, 1 = disease present) for simplicity
y = (y > 0).astype(int).values.ravel()

print("Splitting and preprocessing data...")
# Split the data properly into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline to impute missing values and train the model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(pipeline, 'triage_model.pkl')
print("Model saved as triage_model.pkl!")