import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = pd.read_csv(r"C:\Users\aweso\OneDrive\Desktop\ML_Project\games.csv")
df = df.dropna() 

df['victory_status'] = LabelEncoder().fit_transform(df['victory_status'])
df['increment_code'] = LabelEncoder().fit_transform(df['increment_code'])
df['winner'] = LabelEncoder().fit_transform(df['winner'])

target = 'white_rating'
features = ['turns', 'victory_status', 'increment_code', 'opening_ply', 'winner']

df['rating_diff'] = df['white_rating'] - df['black_rating']
df['did_win'] = (df['winner'] == 1).astype(int)  
features += ['rating_diff', 'did_win']

X = df[features]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),  
    ('scaler', StandardScaler()),  
    ('ridge', Ridge()) 
])

param_grid = {
    'ridge__alpha': [0.1, 1, 10, 100]  
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best alpha parameter for Ridge regression: {grid_search.best_params_['ridge__alpha']}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")
