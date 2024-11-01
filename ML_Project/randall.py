import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")


df = df.dropna()
df['winner'] = LabelEncoder().fit_transform(df['winner'])

features = ['white_rating', 'black_rating', 'turns', 'opening_ply']
X = df[features]
y = df['winner']

X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_training, y_training)

y_predictions = model.predict(X_testing)
accuracy = accuracy_score(y_testing, y_predictions)
report = classification_report(y_testing, y_predictions, target_names=['black','draw', 'white'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)




