
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title('Tractor Sales Forecasting with SARIMAX')

# --- Load the Model ---
@st.cache_resource
def load_model():
    model = joblib.load('arima_model.pkl')
    return model

model = load_model()

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/akhiljamdar/Sales-forecasting-using-Time-series-analysis/refs/heads/master/Tractor-Sales.csv")
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format='%b-%y')
    df.set_index("Month-Year", inplace=True)
    df1 = df["Number of Tractor Sold"]
    return df1

df1 = load_data()

# Define test_no (should match training split)
test_no = 24

train_data = df1[:-test_no]
test_data = df1[-test_no:]

# --- Make Predictions ---
Y_hat = model.predict(n_periods=test_no)
Y_hat.index = test_data.index # Ensure the predictions have the correct index

# --- Visualization ---
st.subheader('Tractor Sales Forecast')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_data, label='Training Data')
ax.plot(test_data, label='Actual Test Data')
ax.plot(Y_hat, label='ARIMA Predictions', linestyle='--')
ax.set_title('Tractor Sales: Actual vs. Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Tractors Sold')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.write("### Prediction Details")
st.write("Here are the forecasted values for the next 24 months:")
st.write(Y_hat)
