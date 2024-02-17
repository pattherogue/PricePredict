import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic

# Load the dataset
df = pd.read_csv('AB_US_2023.csv')

# Brainstorm features
# Example: Calculate distance from city center
city_center_coords = (city_center_latitude, city_center_longitude)  # Coordinates of the city center
property_coords = list(zip(df['latitude'], df['longitude']))  # Coordinates of properties
distances_from_city_center = [geodesic(property_coords[i], city_center_coords).kilometers for i in range(len(df))]
df['distance_from_city_center_km'] = distances_from_city_center

# Split the dataset into training and test sets
X = df.drop('price', axis=1)  # Features (all columns except 'price')
y = df['price']  # Target variable ('price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, columns=['property_type'])
label_encoder = LabelEncoder()
df['host_id'] = label_encoder.fit_transform(df['host_id'])

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
