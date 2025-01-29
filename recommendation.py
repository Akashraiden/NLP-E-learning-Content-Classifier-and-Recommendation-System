import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import re

with open('model_sub_cat.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizers.pkl', 'rb') as vec_file:
    loaded_data = pickle.load(vec_file)

vectorizer = loaded_data['vectorizer']
le_course_type = loaded_data['label_encoder_course_type']
le_sub_cat = loaded_data['label_encoder_sub_category']

df = pd.read_csv('df_dataset.csv')
df.dropna(inplace=True)

X_title = vectorizer.transform(df['Title'])
X_skills = vectorizer.transform(df['Skills'])

df['Course Type'] = le_course_type.transform(df['Course Type'])

def extract_duration_in_months(duration_text):
    match = re.search(r'(\d+)\s*months', str(duration_text))
    return int(match.group(1)) if match else 0

df['Duration'] = df['Duration'].apply(extract_duration_in_months)

X_course_type = df[['Course Type']].values
X_duration = df[['Duration']].values
X = hstack([X_title, X_skills, X_course_type, X_duration])

def recommend_similar_courses(course_idx, top_n=3):
    cosine_sim = cosine_similarity(X, X)
    sim_scores = list(enumerate(cosine_sim[course_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    course_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[course_indices][['Title', 'Sub-Category', 'Duration']]
    return recommendations

recommended_courses = recommend_similar_courses(0, top_n=3)

print("Recommended Courses:")
print(recommended_courses)
