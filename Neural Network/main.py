# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import os
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

os.chdir("/content/drive/My Drive/ECE ML/Week 2/PW_2")
!ls

"""1.1. Load the csv file `data.csv` into a dataframe called `df` and print its shape. (Set the right parameters when reading the csv file)"""

df = pd.read_csv('Assignment_2_data.csv',index_col=0)
df.shape

# check if your answer is correct
assert df.shape == (18207, 88)

"""1.2. print the head of `df`"""

df.head(2)

"""1.3. Print how many columns that are in df columns types"""

df.dtypes.value_counts()

"""1.4. `to_drop` is a list containing columns that are not useful for modeling, remove them and print the new shape of `df`"""

to_drop =['ID', 'Name', 'Photo','Nationality', 'Flag','Club','Club Logo', 'Real Face', 'Joined', 'Loaned From', 'Contract Valid Until']
df.drop(to_drop, axis = 1, inplace = True)
df.shape

# check if your answer is correct
assert df.shape == (18207, 77)

"""# Data Cleaning

## Handling missing values

2.1. Build a dataframe called `missing` which has the following format:

* `pct` is the percentage of missing values, **takes values between `0` and `100`**
* the index is the column names

|     | pct |
|-----|-----|
|......|.....|
|Strength |0.263635|
|.....|.....|
"""

missing = pd.DataFrame(df.isnull().sum() / len(df) * 100, columns=['pct'])

"""2.2. Remove from `missing`, rows with `pct`= 0   
sort `missing` in ascending order of `pct` and print its head
"""

missing = missing[missing['pct'] != 0]
missing.sort_values('pct', inplace = True)
missing.head()

"""2.3. Now, let's fill missing values where the % of missing is lower than 1 (1%).   
First identify these columns in a list named `cols_to_fill`
"""

threshold_missing_to_fill = 1
cols_to_fill = list(missing[missing['pct'] < threshold_missing_to_fill].index)
print(len(cols_to_fill), type(cols_to_fill))

# check if your answer is correct
assert len(cols_to_fill) == 44; assert isinstance(cols_to_fill, list)

"""2.4. define a function (`fill_nas_by_type`) to fill null values by column type:

* if a column type is `Object`, fill it with the **most frequent value**
* otherwise, fill it with the **median value**
"""

def fill_nas_by_type(df, col_name):
    """Fill null values in df according to col_name type
    
    Parameters
    ----------
    df : dataframe, (default=None)
        input dataframe
    col_name : str, (default=None)
        column with null values to fill
        
    Returns
    -------
    df with filled values in col_name
    """
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

"""Loop through `cols_to_fill` and apply the defined function `fill_nas_by_type` to fill null values"""

for f in cols_to_fill:
    df = fill_nas_by_type(df.copy(), col_name = f)

# check if your answer is correct
assert df[cols_to_fill].isnull().sum().sum() == 0

"""For the remaining missing values, let's just remove them.    
Print the shape of `df` before and after removing any rows with missing observations
"""

print(df.shape)
df.dropna(axis = 0, inplace = True)
print(df.shape)

# check if your answer is correct
assert df.shape == (14743, 77); assert df.isnull().sum().sum() == 0

"""## Correct some columns format

### Monetary columns
"""

money_cols = ['Value','Wage', 'Release Clause']
df[money_cols].head()

"""3.1. Build a function which extracts the monetary value from a string. It should return a number with no decimals.   
Your function should pass the three tests in the cell after
"""

def get_value(value_text):
    """Extract the monetary value from a string
    
    Parameters
    ----------
    value_text: str, (default=None)
        a string containing a number ending with M, K or nothing
        
    Returns
    -------
    a float with no decimals
    
    Examples
    --------
    >>> get_value('€7.1K')
    7100.0
    """
    multiplier = value_text[-1]
    if multiplier == 'M':
        number = float(value_text[1:-1])
        return number * 1000000
    elif multiplier == 'K':
        number = float(value_text[1:-1])
        return number * 1000
    else:
        return float(value_text[1:])

# check if your answer is correct
assert get_value('€110.5M') == 110500000; assert get_value('€7.1K') == 7100; assert get_value('€200') == 200

"""3.2. Loop through `money_cols` and apply the defined function `get_value` to convert them to numeric"""

for f in money_cols:
    df[f] =df[f].apply(get_value)
    print(f, df[f].dtype, df[f].isnull().sum())

# check if your answer is correct
assert df[money_cols].isnull().sum().sum() == 0
print(df[money_cols])

"""### Height and Weight columns

4.1. Start by printing the unique values for `Height`
"""

# print unique values for Height
df['Height'].unique()

"""4.2. Write a function (`get_height`) which converts the Height from a string in feet to  a number in `cm` with no decimals.    
1 feet = 30.48 cm. For example `get_height("5'10")` = `155`
"""

def get_height(x):
    return  round(float(x.replace("'", ".")) * 30.48, 0)

# check if your answer is correct
assert get_height("5'10") == 155; assert get_height("6'8") == 207

"""Apply the previous defined function on `Height`"""

df['Height'] = df['Height'].apply(get_height)

# check if your answer is correct
assert df['Height'].dtype == 'float64'; assert df['Height'].isnull().sum() == 0

"""4.3. The same thing with `Weight`, print the unique values"""

# print unique values for Weight
df['Weight'].unique()

"""4.4. Write a function (`get_weight`) which converts the **Weight** from a string in `lbs` to a number in `kg` with no decimals.    
1 lbs = 0.453592 kg. For example `get_weight("115lbs")` = `52`
"""

def get_weight(x):
    return  round(float(x.split('lbs')[0]) * 0.453592, 0)

# check if your answer is correct
assert get_weight("115lbs") == 52; assert get_weight("234lbs") == 106

"""Apply the previous defined function on `Weight`"""

df['Weight'] = df['Weight'].apply(get_weight)

# check if your answer is correct
assert df['Weight'].dtype == 'float64'; assert df['Weight'].isnull().sum() == 0

df["Height"].head()

"""## Convert text columns to numeric

5.1. Identify non-numeric text columns in a list called `text_cols`
"""

text_cols = [f for f in df.columns if df[f].dtype == 'O']
print(len(text_cols))

"""5.2. Build a list named `cols_to_remove` containing columns from `text_cols`, if a column has a number of unique values greater than **10** (`> 10`)"""

cols_to_remove = []
threshold_too_many_unics = 10
for f in text_cols:
    if df[f].nunique() > threshold_too_many_unics:
        cols_to_remove.append(f)
print(len(cols_to_remove))

"""remove `cols_to_remove` columns from `df` and print its shape"""

df.drop(cols_to_remove, axis = 1, inplace = True)
df.shape

# check if your answer is correct
assert df.shape == (14743, 50)

"""5.3. Identify the remaining text columns in `text_cols` as `remaining_text_cols`, make sur it passes the test after"""

remaining_text_cols = [f for f in df.columns if df[f].dtype == 'O']
print(len(remaining_text_cols))

# check if your answer is correct
assert remaining_text_cols == ['Preferred Foot', 'Work Rate', 'Body Type']

"""5.4. Loop through `remaining_text_cols` and convert them to numerical values"""

for f in remaining_text_cols:
    df[f]= df[f].astype("category").cat.codes

df.shape

# df.to_csv('Assignment_2_data_cleaned.csv', index = False)

"""# Model building

As stated before, you can do this part without completing the previous one

6.1. Load the cleaned dataset `Assignment_2_data_cleaned.csv` into `df_clean` and print its shape.
"""

df_clean = pd.read_csv('Assignment_2_data_cleaned.csv')
df_clean.shape

"""6.2. Load the target variable `Overall` into a dataframe and name it `y`. Then, load the features into a second dataframe and name it `X`. Plot a histogram of `y`, choose the number of bins as 100."""

y = df_clean.Overall
X = df_clean.drop('Overall', axis=1, inplace=False)
y.hist(bins=100)

"""7. Split the data set into a training set and a test set. Choose `test_size` = 0.3 and `random_state` = 123  
Print train and test size      
**Attention**: You are asked to use  [`sklearn.model_selection`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
"""

from sklearn.model_selection import train_test_split
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
print("train set shape: ", X_train.shape, y_train.shape)
print("test set shape: ", X_test.shape, y_test.shape)

"""8. Fit a linear model to the training set, and then report the training and testing errors obtained (the R2 statistic).   
Calculate and print the following metrics: mse, rmse, mae for the test_set
"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
train_score = reg.score(X_train, y_train)
test_score=reg.score(X_test, y_test)
print  ('train score =', train_score)
print  ('test score = {}'.format(test_score))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print ('mse = {}, rmse = {} \nmae = {} r2 = {}'.format(mse,math.sqrt(mse), mae, r2))

"""### Check residuals

9.1. Plot a histogram of the residuals (difference between `y_test` and `y_pred`
"""

plt.figure(figsize= (8, 3))
plt.hist(y_test - y_pred)
plt.xlabel('residuals')
plt.ylabel('')
plt.show()

"""9.2. Plot a scatter plot where `y_test` is in the **x** axis and  `y_pred` is in the **y** axis"""

plt.figure(figsize= (8, 3))
plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

"""10. Try to improve the performance of your model, by adding new features"""

X['wage_ratio'] = X['Wage'] / X['Value']

value_by_age = X.groupby('Age')['Value'].median().to_dict()
X['value_by_age_median'] = X['Age'].map(value_by_age)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
reg=LinearRegression()
reg.fit(X_train, y_train)
train_score = reg.score(X_train, y_train)
test_score=reg.score(X_test, y_test)
print  ('train score =', train_score)
print  ('test score = {}'.format(test_score))

#print(reg.coef_)
index_to_predict = 0
print("Value to predict is ",y[index_to_predict])

print(reg.predict(np.array(X.iloc[index_to_predict]).reshape(1,-1))[0])

