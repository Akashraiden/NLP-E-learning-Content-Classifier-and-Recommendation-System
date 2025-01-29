import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('df_dataset.csv')
df.dropna(inplace=True)

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def extract_duration_in_months(duration_text):
    match = re.search(r'(\d+)\s*months', duration_text)
    return int(match.group(1)) if match else 0

df['Duration'] = df['Duration'].apply(extract_duration_in_months)
df['Title'] = df['Title'].apply(clean_text)
df['Skills'] = df['Skills'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')

X_title = vectorizer.fit_transform(df['Title'])
X_skills = vectorizer.transform(df['Skills'])

le_course_type = LabelEncoder()
df['Course Type'] = le_course_type.fit_transform(df['Course Type'])

le_sub_cat = LabelEncoder()
df['Sub-Category'] = le_sub_cat.fit_transform(df['Sub-Category'])

sub_category_mapping = dict(zip(le_sub_cat.transform(le_sub_cat.classes_), le_sub_cat.classes_))
course_type_mapping = dict(zip(le_course_type.transform(le_course_type.classes_), le_course_type.classes_))

X_course_type = df[['Course Type']].values
X_duration = df[['Duration']].values
X = hstack([X_title, X_skills, X_course_type, X_duration])

y_sub_cat = df['Sub-Category']

X_train, X_test, y_train_sub_cat, y_test_sub_cat = train_test_split(X, y_sub_cat, test_size=0.2, random_state=42)

model_sub_cat = RandomForestClassifier(class_weight='balanced', random_state=42)
model_sub_cat.fit(X_train, y_train_sub_cat)

y_pred_sub_cat = model_sub_cat.predict(X_test)

print(f"Accuracy for Sub-Category: {accuracy_score(y_test_sub_cat, y_pred_sub_cat)}")
print(f"Precision for Sub-Category: {precision_score(y_test_sub_cat, y_pred_sub_cat, average='weighted')}")
print(f"Recall for Sub-Category: {recall_score(y_test_sub_cat, y_pred_sub_cat, average='weighted')}")
print(f"F1-Score for Sub-Category: {f1_score(y_test_sub_cat, y_pred_sub_cat, average='weighted')}")

title_to_predict = "introduction data science specialization"
skills_to_predict = "Python"
course_type_to_predict = "Specialization"
duration_to_predict = "Approximately 4 months to complete"

title_to_predict_cleaned = clean_text(title_to_predict)
skills_to_predict_cleaned = clean_text(skills_to_predict)
duration_to_predict_numeric = extract_duration_in_months(duration_to_predict)

if course_type_to_predict not in course_type_mapping.values():
    print(f"Error: '{course_type_to_predict}' is not a recognized course type.")
else:
    course_type_to_predict_encoded = le_course_type.transform([course_type_to_predict])
    title_vectorized = vectorizer.transform([title_to_predict_cleaned])
    skills_vectorized = vectorizer.transform([skills_to_predict_cleaned])
    X_predict = hstack([title_vectorized, skills_vectorized, [[course_type_to_predict_encoded[0]]], [[duration_to_predict_numeric]]])
    predicted_sub_category = model_sub_cat.predict(X_predict)
    predicted_sub_category_label_name = sub_category_mapping[predicted_sub_category[0]]
    print(f"Predicted Sub-Category for '{title_to_predict}' with Course Type '{course_type_to_predict}' and Duration '{duration_to_predict}': {predicted_sub_category_label_name}")

import pickle

with open('model_sub_cat.pkl', 'wb') as model_file:
    pickle.dump(model_sub_cat, model_file)

with open('vectorizers.pkl', 'wb') as vec_file:
    pickle.dump({
        'vectorizer': vectorizer,
        'label_encoder_course_type': le_course_type,
        'label_encoder_sub_category': le_sub_cat
    }, vec_file)

print("Pickle files saved successfully!")
