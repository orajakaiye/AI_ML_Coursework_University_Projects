#!/usr/bin/env python
# coding: utf-8

# In[479]:


# Importing libraries for linear algebra, data processing, and visualization
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting

# Importing libraries from sklearn for model creation and evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # For creating the regression model
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation


# In[480]:


# Load the dataset
houseprice_data = pd.read_csv('houseprice_data.csv') 
# Basic exploration of the data
print(houseprice_data.head())  # Display the first 5 rows of the dataset
print(houseprice_data.tail())  # Display the last 5 rows of the dataset
print(houseprice_data.describe())  # Summary statistics (mean, std, min, max, etc.)
print(houseprice_data.info())  # Info about the DataFrame (types, non-null counts)
print(houseprice_data.corr(), '\n')  # Correlation matrix to understand feature relationships


# In[481]:


# Define your features (X) and target variable (y)
X = houseprice_data[['bedrooms', 'bathrooms', 'sqft_living', 'waterfront', 'view', 'grade', 'yr_built', 'lat']]  # Features used for prediction
y = houseprice_data['price'].values  # Target variable (house price)

# Display the features (X) and target variable (y)
print(X)
print(y)


# In[482]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=75)

# Confirm the split
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# In[483]:


# Create a LinearRegression model and fit it to the training data
regr = LinearRegression()
regr.fit(X_train, y_train)


# In[498]:


# Loop through each selected feature to create scatter plots
for feature in X.columns:
    fig, ax = plt.subplots()
    ax.scatter(houseprice_data[feature], y, color='blue')  # Scatter plot of feature vs price
    
    # Set labels and title for each plot
    ax.set_xlabel(feature)  # x-axis label is the feature name
    ax.set_ylabel('Price')  # y-axis label is 'Price'
    ax.set_title(f'Scatter Plot of {feature} vs Price')  # Title for each plot
    
    # Adjust the layout and save the figure
    fig.tight_layout()
    fig.savefig(f'Houseprices_{feature}_vs_Price_plot.png')  # Save the plot as an image file


# In[499]:


# Make predictions using the test set
y_pred = regr.predict(X_test)

# Print model coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Evaluate the model performance
print('Mean squared error: %.8f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R²): %.2f' % r2_score(y_test, y_pred))


# In[505]:


# Visualize train set results
fig, ax = plt.subplots(figsize=(9, 5))

# Scatter plot of actual vs predicted house prices (train set)
ax.scatter(y_train, regr.predict(X_train), color='red')  # Actual prices vs predicted prices
ax.plot(y_train, y_train, color='blue')  # Perfect prediction line
ax.set_xlabel('Combined Features')
ax.set_ylabel('Price')
ax.set_title('Scatter Plot of Combined Features vs Price (Training Set)')
fig.savefig('Combined_Features_vs_Price_Training.png')
# Add legend

# Show the plots
plt.show()

# Adjust layout and save the plot
fig.tight_layout()
fig.savefig('Houseprices_trainingset_plot.png')  # Save the plot
plt.show()


# In[503]:


# Visualize test set results
fig, ax = plt.subplots(figsize=(9, 5))

# Scatter plot of actual vs predicted house prices (test set)
ax.scatter(y_test, regr.predict(X_test), color='red')  # Actual prices vs predicted prices
ax.plot(y_test, y_test, color='blue')  # Perfect prediction line
ax.set_xlabel('Combined Features')
ax.set_ylabel('Price')
ax.set_title('Scatter Plot of Combined Features vs Price (Testing Set)')
fig.savefig('Combined_Features_vs_Price_Test.png')

# Show the plots
plt.show()

# Adjust layout and save the plot
fig.tight_layout()
fig.savefig('Houseprices_testset_plot.png')  # Save the plot
plt.show()


# ## Author: Oluwadamilola Oreofe Ajakaiye
# ## Data Science Consultant
# ### Date: 16th October, 2024

# In[ ]:




