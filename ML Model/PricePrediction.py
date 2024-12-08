import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# Load the dataset
data = pd.read_csv("Housing.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data["mainroad"] = label_encoder.fit_transform(data["mainroad"])  # yes/no to 1/0
data["guestroom"] = label_encoder.fit_transform(data["guestroom"])  # yes/no to 1/0
data["basement"] = label_encoder.fit_transform(data["basement"])  # yes/no to 1/0
data["hotwaterheating"] = label_encoder.fit_transform(data["hotwaterheating"])  # yes/no to 1/0
data["airconditioning"] = label_encoder.fit_transform(data["airconditioning"])  # yes/no to 1/0
data["prefarea"] = label_encoder.fit_transform(data["prefarea"])  # yes/no to 1/0
data["furnishingstatus"] = label_encoder.fit_transform(data["furnishingstatus"])  # furnished/semi-furnished/unfurnished

# Prepare features and target
X = data.drop("price", axis=1)  # Features (exclude price)
y = data["price"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

# Predict for new data
new_data = [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 0]]  # Example input
# Convert new_data to DataFrame to match feature names
new_data_df = pd.DataFrame(new_data, columns=X.columns)
predicted_price = model.predict(new_data_df)

print(f"Predicted Price: ${predicted_price[0]:,.2f}")

# Save the trained model
joblib.dump(model, "linear_regression_model.pkl")
print("Model saved as linear_regression_model.pkl")