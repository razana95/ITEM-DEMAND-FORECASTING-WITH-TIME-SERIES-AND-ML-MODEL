import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX

df=pd.read_csv("train 2.csv",parse_dates=["date"],index_col=["date"])

st.title('SARIMAX Sales Forecasting App')

# Sidebar Inputs
selected_item = st.selectbox('Select Item:', df['item'].unique())
forecast_periods = st.slider('Select Forecast Periods:', 1, 12, 3)

# Load the saved model for the selected item
with open(f'sarimax_model_{selected_item}.pkl', 'rb') as f:
    saved_model = pickle.load(f)

# Example: Forecast sales using the saved model
forecast = saved_model.forecast(steps=forecast_periods)
forecast_dates = pd.date_range(start=df.index.max(), periods=forecast_periods + 1, freq='M')[1:]

# Display forecast results as a dataframe
forecast_df = pd.DataFrame({'Forecast Months': forecast_dates, 'Sales Forecast in Rupee': round(forecast,2)})
st.write(f"Forecast for {forecast_periods} months for item {selected_item}:")
st.write(forecast_df)

if st.button(f"Forecasting plot for item {selected_item}"):
    item_data = df[df['item'] == selected_item]
    monthly_sales = item_data.resample('M').sum()

    # Fit the model to the entire data
    model = SARIMAX(monthly_sales['sales'], seasonal_order=(0, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Predict sales using the model
    predicted_sales = model_fit.predict(start=len(monthly_sales), end=len(monthly_sales) + forecast_periods - 1)

    # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(monthly_sales.index, monthly_sales['sales'], label='Actual Sales')
    
    # plt.plot(forecast_dates, forecast, label='Forecasted Sales', linestyle='dashed')
    # #plt.plot(forecast_dates, predicted_sales, label='Predicted Sales', linestyle='dashed')
    # plt.xlabel('Date')
    # plt.ylabel('Sales')
    # plt.title(f'Sales Forecast and Prediction for {selected_item}')
    # plt.legend()
    # plt.xticks(rotation=45)
    # st.pyplot(plt)
  
    fig = px.line(monthly_sales, x=monthly_sales.index, y='sales', title=f'Sales Forecast and Prediction for {selected_item}')
    fig.add_trace(px.line(x=forecast_df['Forecast Months'], y=forecast).data[0].update(name='Forecasted Sales'))
    #fig.add_trace(px.line(x=forecast_df['Forecast Months'], y=predicted_sales).data[0].update(name='Predicted Sales'))

        # Update the layout to include hover mode and show data on hover
    fig.update_layout(hovermode='x unified')

        # Show the plot
    st.plotly_chart(fig)

    