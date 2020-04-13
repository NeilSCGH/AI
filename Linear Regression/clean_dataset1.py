import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##Getting data
#Reading csv
df = pd.read_csv('dataset1.csv',index_col=0)
#df.head(2)
#df.dtypes.value_counts() #number of column

#Checking if the data size is correct
assert df.shape == (18207, 88)

#Dropping useless columns
to_drop =['ID', 'Name', 'Photo','Nationality', 'Flag','Club','Club Logo', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until']
df.drop(to_drop, axis = 1, inplace = True)

#Checking if the data size is correct
assert df.shape == (18207, 77)

##Data Cleaning
# Handling missing values
#* pct is the percentage of missing values
#* the index is the column names
missing = pd.DataFrame(df.isnull().sum() / len(df) * 100, columns=['pct'])

#Removing from missing, rows with pct= 0
missing = missing[missing['pct'] != 0]
missing.sort_values('pct', inplace = True)

#Filling missing values where the % of missing is lower than 1 (1%).
threshold_missing_to_fill = 1
cols_to_fill = list(missing[missing['pct'] < threshold_missing_to_fill].index)

#Checking if the data size is correct
assert len(cols_to_fill) == 44
assert isinstance(cols_to_fill, list)

#Defining a function to fill null values by column type:
#if a column type is Object, fill it with the most frequent value
#otherwise, fill it with the median value
def fill_nas_by_type(df, col_name):
    if df[col_name].isnull().sum() == 0:
        print('!! Warning : %s does not have null values' % col_name)
        return df
    if df[col_name].dtype == 'O':
        fill_value = df[col_name].value_counts().index[0]
        df[col_name].fillna(fill_value, inplace = True)
    else:
        fill_value = df[col_name].median()
        df[col_name].fillna(fill_value, inplace = True)
    return df

#Applying the function to fill null values
for f in cols_to_fill:
    df = fill_nas_by_type(df.copy(), col_name = f)

#Checking if the data size is correct
assert df[cols_to_fill].isnull().sum().sum() == 0

#Removing remaining missing values
df.dropna(axis = 0, inplace = True)

#Checking if the data size is correct
assert df.shape == (14743, 77)
assert df.isnull().sum().sum() == 0

#Correcting some columns format
#Monetary columns
money_cols = ['Value','Wage', 'Release Clause']

#Building a function which extracts the monetary value from a string. It should return a number with no decimals
def get_value(value_text):
    multiplier = value_text[-1]
    if multiplier == 'M':
        number = float(value_text[1:-1])
        return number * 1000000
    elif multiplier == 'K':
        number = float(value_text[1:-1])
        return number * 1000
    else:
        return float(value_text[1:])

#Checking if the function works correctly
assert get_value('€110.5M') == 110500000
assert get_value('€7.1K') == 7100
assert get_value('€200') == 200

#Applying the function
for f in money_cols:
    df[f] =df[f].apply(get_value)

#Checking
assert df[money_cols].isnull().sum().sum() == 0

# Height and Weight columns
#Converting the Height from a string in feet to  a number in cm with no decimals.
def get_height(x):
    return  round(float(x.replace("'", ".")) * 30.48, 0)

#Checking if the function works correctly
assert get_height("5'10") == 155
assert get_height("6'8") == 207

#Applying the function
df['Height'] = df['Height'].apply(get_height)

#Checking
assert df['Height'].dtype == 'float64'
assert df['Height'].isnull().sum() == 0

#Same thing with Weight
#Converting the Weight from a string in lbs to a number in kg with no decimals.
def get_weight(x):
    return  round(float(x.split('lbs')[0]) * 0.453592, 0)

#Checking if the function works correctly
assert get_weight("115lbs") == 52
assert get_weight("234lbs") == 106

#Applying the function
df['Weight'] = df['Weight'].apply(get_weight)

#Checking
assert df['Weight'].dtype == 'float64'
assert df['Weight'].isnull().sum() == 0


#Converting text columns to numeric
text_cols = [f for f in df.columns if df[f].dtype == 'O']

#Building a list named cols_to_remove containing columns from text_cols if a column has a number of unique values greater than 10
cols_to_remove = []
threshold_too_many_unics = 10
for f in text_cols:
    if df[f].nunique() > threshold_too_many_unics:
        cols_to_remove.append(f)

#Drop them
df.drop(cols_to_remove, axis = 1, inplace = True)

#Checking if the data size is correct
assert df.shape == (14743, 50)

#Identifying the remaining text columns in text_cols as remaining_text_cols
remaining_text_cols = [f for f in df.columns if df[f].dtype == 'O']

#Checking
assert remaining_text_cols == ['Preferred Foot', 'Work Rate', 'Body Type']

#Looping through remaining_text_cols and convert them to numerical values
for f in remaining_text_cols:
    df[f]= df[f].astype("category").cat.codes

df.to_csv('data_cleaned.csv', index = False)
