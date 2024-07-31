import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
params = {'vs_currency': 'usd', 'days': '365'}
response = requests.get(url, params=params)
data = response.json()
prices = data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.to_csv('bitcoin_data.csv', index=False)

def visualize_data(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

    sns.lineplot(x='timestamp', y='price', data=df)
    plt.title('Bitcoin Price Over Time')
    plt.show()

    fig = px.line(df, x='timestamp', y='price', title='Bitcoin Price Over Time')
    fig.show()

visualize_data(df)

df['price_shifted'] = df['price'].shift(-1)
df = df.dropna()
X = np.array(df['price']).reshape(-1, 1)
y = np.array(df['price_shifted']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
}

best_model = None
best_score = float('inf')
for name, model in models.items():
    model.fit(X_train, y_train.ravel())
    predictions = model.predict(X_test)
    score = mean_squared_error(y_test, predictions)
    print(f'{name} MSE: {score}')
    if score < best_score:
        best_score = score
        best_model = model

print(f'Best model: {best_model}')

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train.ravel())
print(f'Best params: {grid_search.best_params_}')

best_predictions = grid_search.predict(X_test)
plt.scatter(y_test, best_predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

print(f'MAE: {mean_absolute_error(y_test, best_predictions)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, best_predictions))}')

from fastapi import FastAPI
app = FastAPI()

@app.post('/predict')
def predict(price: float):
    prediction = grid_search.predict([[price]])
    return {'predicted_price': prediction[0]}
