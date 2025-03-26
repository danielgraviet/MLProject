import numpy as np # handels math and arrays
import pandas as pd # spreadsheet for data. 
import matplotlib.pyplot as plt # makes graphs and plots
from sklearn.linear_model import LinearRegression # model for ML, uses a straight line. 
from sklearn.model_selection import train_test_split # splits data into training and testing parts. 

# some basic dictionary data, can be replaced with a larger data CSV file in the future
data = {
    'Size': [600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    'Price': [150000, 200000, 240000, 280000, 350000, 400000, 450000, 480000, 550000, 650000]
}

# turn the data into a dataFram which is a table-like structure
dataFrame = pd.DataFrame(data) 

# prepare the data for the model
X = dataFrame[['Size']] # why is this capitolized? 
y = dataFrame['Price']

# split data into training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# make a prediction
y_prediction = model.predict(X_test)

# print the results
print("Test set predictions:")
for size, pred_price in zip(X_test['Size'], y_prediction):
    print(f"House size: {size} sq ft, Predicted price: ${pred_price:.2f}")
    
# visualize the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_prediction, color='red', label='Predicted Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)

# predict a new house price
new_size = pd.DataFrame([[1400]], columns=['Size'])
new_price = model.predict(new_size)
print(f"Predicted price for a 1400 sq ft house: ${new_price[0]:.2f}")

plt.show(block=True)