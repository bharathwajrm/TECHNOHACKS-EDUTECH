
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv("kc_house_data.csv")


X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
          'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
          'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import joblib
joblib.dump(model, 'trained_model.pkl')
