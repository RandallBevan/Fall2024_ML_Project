import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")
df = df.dropna()

df['winner'] = LabelEncoder().fit_transform(df['winner'])
df['victory_status'] = LabelEncoder().fit_transform(df['victory_status'])
df['increment_code'] = LabelEncoder().fit_transform(df['increment_code'])

features = ['white_rating', 'black_rating', 'turns', 'opening_ply', 'victory_status', 'increment_code']
X = df[features]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
y = df['winner']

X_training, X_testing, y_training, y_testing = train_test_split(X_poly, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200))
])

pipeline.fit(X_training, y_training)

y_predictions = pipeline.predict(X_testing)
accuracy = accuracy_score(y_testing, y_predictions)
report = classification_report(y_testing, y_predictions, target_names=['black', 'draw', 'white'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
