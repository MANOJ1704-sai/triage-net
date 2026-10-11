import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Loading MTSamples dataset...")
df = pd.read_csv('mtsamples.csv')

# Drop empty transcription rows
df = df.dropna(subset=['transcription'])

# For the hackathon, we will derive 'urgency' from medical specialties.
# High risk: Surgery, Emergency Room Reports, Cardiovascular / Pulmonary
critical_keywords = ['Surgery', 'Emergency', 'Cardiovascular', 'Pulmonary']
df['is_urgent'] = df['medical_specialty'].apply(
    lambda x: 1 if any(keyword in str(x) for keyword in critical_keywords) else 0
)

X = df['transcription']
y = df['is_urgent']

print("Splitting and preprocessing text data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline with a TF-IDF vectorizer and Logistic Regression baseline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42))
])

print("Training clinical text model...")
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, 'text_model.pkl')
print("Model saved as text_model.pkl!")
