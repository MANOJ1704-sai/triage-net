import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

print("Loading Pima Indians Diabetes dataset...")
# Define the standard columns for this dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("pima-indians-diabetes.csv", names=columns)

# In this dataset, a value of 0 in certain columns actually means 'missing data'
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

print("Training diabetes model...")
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'diabetes_model.pkl')
print("Model saved as diabetes_model.pkl!")