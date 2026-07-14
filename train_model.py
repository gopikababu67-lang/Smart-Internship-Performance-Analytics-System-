import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

root = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(root, 'data', 'combined_dataset_FINAL_CLEAN.xlsx')
model_dir = os.path.join(root, 'models')
os.makedirs(model_dir, exist_ok=True)

df = pd.read_excel(data_path)
branch_categories = list(pd.unique(df['branch']))
college_categories = list(pd.unique(df['college_tier']))

mapping = {'Male': 0, 'Female': 1, 'Other': 2}
X = pd.DataFrame({
    'age': df['age'].astype(float),
    'gender': df['gender'].map(mapping).fillna(2).astype(int),
    'cgpa': df['cgpa'].astype(float),
    'branch': df['branch'].map({v: i for i, v in enumerate(branch_categories)}).astype(int),
    'college_tier': df['college_tier'].map({v: i for i, v in enumerate(college_categories)}).astype(int),
    'internships': df['internships_count'].astype(float),
    'projects': df['projects_count'].astype(float),
    'certifications': df['certifications_count'].astype(float),
    'coding_score': df['coding_skill_score'].astype(float),
    'aptitude': df['aptitude_score'].astype(float),
    'communication': df['communication_skill_score'].astype(float),
    'hackathons': df['hackathons_participated'].astype(float),
    'github_repos': df['github_repos'].astype(float),
    'mock_interview': df['mock_interview_score'].astype(float),
    'attendance': df['attendance_percentage'].astype(float),
    'backlogs': df['backlogs'].astype(float),
    'overall_skill': df['overall_skill_score'].astype(float),
    'placement_readiness': df['placement_readiness'].astype(float),
})

y = df['placed_flag'].astype(int)

feature_cols = [
    'age', 'gender', 'cgpa', 'branch', 'college_tier',
    'internships', 'projects', 'certifications',
    'coding_score', 'aptitude', 'communication',
    'hackathons', 'github_repos', 'mock_interview',
    'attendance', 'backlogs', 'overall_skill', 'placement_readiness'
]

X = X[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print('Test accuracy:', acc)
print(classification_report(y_test, pred))

joblib.dump(model, os.path.join(model_dir, 'placement_model.pkl'))
joblib.dump(feature_cols, os.path.join(model_dir, 'feature_columns.pkl'))
print('Saved placement_model.pkl and feature_columns.pkl to', model_dir)
