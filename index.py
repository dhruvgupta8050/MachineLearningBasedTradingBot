import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.metrics import accuracy_score

# Download historical data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
data['Return'] = data['Close'].pct_change()
data['Direction'] = np.where(data['Return'] > 0, 1, 0)

# Load dataset from Kaggle
file_path = "algo-trading-data-nifty-100-data-with-indicators.csv"
kagglehub.authenticate()  # Ensure you are authenticated with Kaggle

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "debashis74017/algo-trading-data-nifty-100-data-with-indicators",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
print("First 5 records:", df.head())
# Feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['RSI_14'] = data['Close'].diff().apply(lambda x: max(x,0)).rolling(14).mean() / \
                 data['Close'].diff().abs().rolling(14).mean()
data = data.dropna()

features = ['SMA_5', 'SMA_10', 'RSI_14']
X = data[features]
y = data['Direction']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
data['Prediction'] = model.predict(X)

# Simple backtest
data['Strategy'] = data['Prediction'].shift(1) * data['Return']
data[['Return', 'Strategy']].cumsum().apply(np.exp).plot(title='Strategy vs Buy & Hold')

# Print accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))