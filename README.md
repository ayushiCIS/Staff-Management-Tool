# Staff-Management-Tool
Utilizing real-world data to advise owners on the appropriate labor allocation for the day of work. Currently used by "The Dhaba", located in Tempe, Arizona!
#Importing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load historical data (Replace with actual dataset)
df = pd.read_csv("restaurant_sales_data.csv")

# Features: Date, Day of the Week, Weather, Events, Previous Sales
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday = 0, Sunday = 6

# Define input features
features = ['DayOfWeek', 'WeatherScore', 'LocalEventsScore', 'PreviousDaySales']
X = df[features]
y = df['OptimalLaborHours']  # Target variable: Recommended labor hours

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a predictive model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict labor hours for the next day
def predict_labor(day_of_week, weather_score, event_score, prev_sales):
    input_data = np.array([[day_of_week, weather_score, event_score, prev_sales]])
    return model.predict(input_data)[0]

# Example usage (Replace with real-time values)
day_of_week = 5  # Saturday
weather_score = 8  # (1-10 scale, 10 being best for business)
event_score = 2  # (Local events impact: 0 = none, 10 = high)
previous_sales = 5000  # Yesterday's revenue

recommended_hours = predict_labor(day_of_week, weather_score, event_score, previous_sales)
print(f"🔹 Recommended Labor Hours: {round(recommended_hours, 2)}")
