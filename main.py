import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(0)
num_samples = 100
data = {
    'Temperature': np.random.uniform(15, 30, num_samples),  # in Celsius
    'Humidity': np.random.uniform(40, 80, num_samples),      # in percentage
    'Turbidity': np.random.uniform(0, 100, num_samples),     # in NTU (Nephelometric Turbidity Units)
    'pH': np.random.uniform(6, 9, num_samples)               # pH level
}

df = pd.DataFrame(data)

# Define features and target
X = df[['Temperature', 'Humidity', 'Turbidity']]
y = df['pH']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R^2 Scores: {cv_scores}")
print(f"Mean Cross-Validation R^2 Score: {cv_scores.mean():.2f}")

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualize the results
plt.figure(figsize=(14, 7))

# Plot actual vs predicted values
plt.subplot(1, 3, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel('Actual pH')
plt.ylabel('Predicted pH')
plt.title('Actual vs Predicted pH')

# Plot residuals
residuals = y_test - y_pred
plt.subplot(1, 3, 2)
sns.histplot(residuals, kde=True, bins=20, color='skyblue')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')

# Plot feature importance
coefficients = model.coef_
features = X.columns
plt.subplot(1, 3, 3)
sns.barplot(x=features, y=coefficients, palette='viridis')
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Feature Importance')

plt.tight_layout()
plt.show()
