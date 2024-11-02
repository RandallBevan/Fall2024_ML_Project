import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")

df = df.dropna()
df['winner'] = LabelEncoder().fit_transform(df['winner'])

features = ['white_rating', 'black_rating', 'turns', 'opening_ply']
X = df[features]
y = df['winner']

X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_training, y_training)

best_model = grid_search.best_estimator_

y_predictions = best_model.predict(X_testing)
accuracy = accuracy_score(y_testing, y_predictions)
report = classification_report(y_testing, y_predictions, target_names=['black', 'draw', 'white'])

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
