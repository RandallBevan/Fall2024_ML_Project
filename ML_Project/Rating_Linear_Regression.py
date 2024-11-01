import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load and preprocess dataset
df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")
df = df.dropna()  # Drop rows with missing values

# Encode categorical features
df['victory_status'] = LabelEncoder().fit_transform(df['victory_status'])
df['increment_code'] = LabelEncoder().fit_transform(df['increment_code'])
df['winner'] = LabelEncoder().fit_transform(df['winner'])

# Define target variable (e.g., white_rating) and relevant features
target = 'white_rating'
features = ['turns', 'victory_status', 'increment_code', 'opening_ply', 'winner']

# Create additional features
df['rating_diff'] = df['white_rating'] - df['black_rating']
df['did_win'] = (df['winner'] == 1).astype(int)  # 1 if white won, 0 otherwise
features += ['rating_diff', 'did_win']

X = df[features]
y = df[target]

# Define the preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), features)  # Impute missing values in numeric features if any
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing, polynomial features, scaling, and Ridge regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),  # Generate polynomial features
    ('scaler', StandardScaler()),  # Scale features
    ('ridge', Ridge())  # Ridge regression for regularization
])

# Define a grid of hyperparameters for tuning
param_grid = {
    'ridge__alpha': [0.1, 1, 10, 100]  # Regularization strength
}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Best alpha parameter for Ridge regression: {grid_search.best_params_['ridge__alpha']}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
