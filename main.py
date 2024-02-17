import pandas as pd
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('AB_US_2023.csv', low_memory=False)

# Identify Columns with Mixed Data Types
mixed_data_columns = ['host_id', 'host_name', 'neighbourhood_group', 'neighbourhood', 'last_review']

# Handle Missing Values
numeric_columns = df.select_dtypes(include=['int', 'float']).columns
numeric_mean = df[numeric_columns].mean()
df.fillna(numeric_mean, inplace=True)

# Specify Data Types
dtype_dict = {'latitude': float, 'longitude': float, 'minimum_nights': int, 'number_of_reviews': int,
              'reviews_per_month': float, 'calculated_host_listings_count': int, 'availability_365': int}
df = pd.read_csv('AB_US_2023.csv', dtype=dtype_dict, low_memory=False)

# Adjust Coordinates (Replace with the coordinates of your city center)
city_center_latitude = 37.8701971329883
city_center_longitude = -122.25944992888442

# Calculate Distance from City Center
city_center_coords = (city_center_latitude, city_center_longitude)
property_coords = list(zip(df['latitude'], df['longitude']))
distances_from_city_center = [geodesic(property_coords[i], city_center_coords).kilometers for i in range(len(df))]
df['distance_from_city_center_km'] = distances_from_city_center

# Choose relevant columns for prediction
selected_columns = ['latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews', 
                    'availability_365', 'calculated_host_listings_count', 'price']
df_selected = df[selected_columns]

# Handle Missing Values (if any)
df_selected.loc[:, df_selected.columns.isin(numeric_mean.index)] = df_selected.loc[:, df_selected.columns.isin(numeric_mean.index)].fillna(numeric_mean)


# Handle Categorical Variables
df_selected = pd.get_dummies(df_selected, columns=['room_type'])

# Split the dataset into training and test sets
X = df_selected.drop('price', axis=1)  # Features (all columns except 'price')
y = df_selected['price']  # Target variable ('price')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
