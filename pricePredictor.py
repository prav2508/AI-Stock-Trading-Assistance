import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Load the historical stock price data from the CSV file
df = pd.read_csv('^GSPC_2010-2015.csv')

reference_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
df['date_numeric'] = (pd.to_datetime(df['Date']) - reference_date).dt.days.astype(float)
train_df = df[:1510]
predDate = df[1510:]
# Extract the date and stock price columns
X = train_df['date_numeric'].values.reshape(-1, 1)
# X = X = X.apply(convertDatetoTimeStamp).values.reshape(-1, 1)
y = train_df['Close'].values.reshape(-1, 1)

# Fit a linear regression model to the data
regressor = LinearRegression()
regressor.fit(X, y)


X_pred = predDate['date_numeric'].values.reshape(-1, 1)
y_pred = regressor.predict(X_pred)

# Print the predicted stock prices for the next 30 days
print('Predicted stock prices for the next 30 days:')


predDate['Pred_Close'] = y_pred

predictedDF = predDate[['Date','Pred_Close']]

predictedDF = predictedDF.rename(columns={"Pred_Close":'Close'})

total = pd.concat([train_df,predictedDF])

total.to_csv('predictedPrice.csv')