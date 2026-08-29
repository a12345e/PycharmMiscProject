import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create some toy data (in real life, you'd load a CSV with pd.read_csv)
data = pd.DataFrame({
    'sqft': [750, 900, 1100, 1300, 1500, 1750, 2000, 2200, 2500, 2800],
    'price': [150000, 180000, 210000, 240000, 270000, 300000, 340000, 370000, 410000, 450000]
})

# 2. Split into features (X) and target (y)
X = data[['sqft']]   # double brackets = keep it as a DataFrame
print(f' {X}')
y = data['price']

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate
print(f"Slope (price per sqft): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² score: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred):.2f}")

# 7. Predict a new value
new_house = pd.DataFrame({'sqft': [1600]})
predicted_price = model.predict(new_house)
print(f"Predicted price for 1600 sqft: ${predicted_price[0]:,.2f}")