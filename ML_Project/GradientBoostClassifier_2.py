import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE  # Import SMOTE from imbalanced-learn

enable_smote = False

# Load dataset
df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")

# Preprocess data
df = df.dropna()
df['winner'] = LabelEncoder().fit_transform(df['winner'])

# Define features and target
features = ['white_rating', 'black_rating', 'turns', 'opening_ply']
X = df[features]
y = df['winner']

# Split data
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE if enabled
if enable_smote:
    smote = SMOTE(random_state=42)
    X_training, y_training = smote.fit_resample(X_training, y_training)

# Create a pipeline for scaling and using Gradient Boosting Classifier
pipeline_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Define parameter grid for hyperparameter tuning
param_grid_gb = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# Grid search
grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_gb.fit(X_training, y_training)

# Best model
best_model_gb = grid_search_gb.best_estimator_

# Make predictions and evaluate
y_predictions_gb = best_model_gb.predict(X_testing)
accuracy_gb = accuracy_score(y_testing, y_predictions_gb)
balanced_accuracy = balanced_accuracy_score(y_testing, y_predictions_gb)
y_testing_binarized = label_binarize(y_testing, classes=[0, 1, 2])
y_predictions_binarized = label_binarize(y_predictions_gb, classes=[0, 1, 2])

# Calculate ROC AUC for each class and average
roc_auc = roc_auc_score(y_testing_binarized, y_predictions_binarized, average='macro', multi_class='ovo')
report_gb = classification_report(y_testing, y_predictions_gb, target_names=['black', 'draw', 'white'])

# Print results
print(f"Best Parameters for Gradient Boosting: {grid_search_gb.best_params_}")
print(f"Accuracy: {accuracy_gb:.2f}")
print(f"Balanced Accuracy: {balanced_accuracy:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
print("Classification Report:\n", report_gb)
