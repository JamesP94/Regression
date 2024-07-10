import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# Set the working directory
os.chdir('C:/Users/james/OneDrive/Documents/Intro to python/regression-tha-JamesP94/data')

# Load the data
hos_data = pd.read_csv('calihospital.txt', delimiter='\t')

# Convert all columns to float
hos_data = hos_data.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype == 'object' else x)

# Create dummy variables for categorical variables
categorical_features = ['Teaching', 'DonorType', 'Gender', 'PositionTitle', 'Compensation', 'TypeControl']
hos_data = pd.get_dummies(hos_data, columns=categorical_features, drop_first=True)

# Bin 'AvlBeds' into categories and convert to dummy variables
hos_data['AvlBeds_Binned'] = pd.cut(hos_data['AvlBeds'], bins=3, labels=["Low", "Medium", "High"])
hos_data = pd.get_dummies(hos_data, columns=['AvlBeds_Binned'], drop_first=True)

# Remove unwanted columns
unwanted_cols = ['HospitalID', 'Name', 'Zip', 'Website', 'NoFTE', 'NetPatRev', 'InOperExp', 'OutOperExp', 'AvlBeds', 
                 'Work_ID', 'LastName', 'FirstName', 'PositionID', 'MaxTerm', 'StartDate']
hos_data.drop(unwanted_cols, axis=1, inplace=True)

# Define the model data
X1 = hos_data.drop(['OperInc', 'OperRev'], axis=1)
X1 = sm.add_constant(X1)  # Add a constant for the intercept
y1 = hos_data['OperInc']

# Fit Model 1
model1 = sm.OLS(y1, X1).fit()

# Fit Model 2
y2 = hos_data['OperRev']
model2 = sm.OLS(y2, X1).fit()

# Output model summaries
print(model1.summary())
print(model2.summary())