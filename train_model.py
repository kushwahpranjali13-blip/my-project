import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load dataset
data = pd.read_csv("fake_news.csv")

print("Loaded dataset successfully!")
print("Columns:", data.columns)
print("Null values before cleaning:\n", data.isnull().sum())

#  Step 2: Fix missing or invalid data
if 'text' not in data.columns:
    raise ValueError("CSV must have a column named 'text'.")

if 'label' not in data.columns:
    raise ValueError("CSV must have a column named 'label'.")

#  Fill NaN values and convert all text to string
data['text'] = data['text'].fillna("").astype(str)
data['label'] = data['label'].fillna("").astype(str)

print("Null values after cleaning:\n", data.isnull().sum())

# Step 3: Prepare features and labels
X = data["text"]
y = data["label"].apply(lambda x: 1 if x.strip().upper() == "REAL" else 0)

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Step 7: Save model and vectorizer
joblib.dump(model, "lr_model.jb")
joblib.dump(vectorizer, "vectorizer.jb")

print("Model and Vectorizer saved successfully!")
