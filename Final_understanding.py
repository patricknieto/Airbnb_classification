
# coding: utf-8

# # Understanding the data

# In any process involving data, the first goal should always be understanding the data. This involves looking at the
# data and answering a range of questions including (but not limited to):
# 
# 1. What features (columns) does the data set contain?
# 2. How many records (rows) have been provided?
# 3. What format is the data in (e.g. what format are the dates provided, are there numerical values, what do the
# different categorical values look like)?
# 4. Are there missing values?
# 5. How do the different features relate to each other?
# 
# 
# We'll start by looking at the 3 main files Airbnb provides us
# 
# 1. train_users_2.csv – This data set contains data on Airbnb users, including the destination countries. Each row
#  represents one user with the columns containing various information such the users’ ages and when they signed up.
# This is the primary data set that we will use to train the model.
# 
# 2. test_users.csv – This data set also contains data on Airbnb users, in the same format as train_users_2.csv,
# except without the destination country. These are the users for which we will have to make our final predictions.
# 
# 3. sessions.csv – This data is supplementary data that can be used to train the model and make the final predictions.
#  It contains information about the actions (e.g. clicked on a listing, updated a  wish list, ran a search etc.)
# taken by the users in both the testing and training data sets above.


import numpy as np
import pandas as pd
import pickle

# test = pd.read_csv('~/../../media/patrick/MY_EXTERNAL/test_users.csv')
train = pd.read_csv('~/../../media/patrick/MY_EXTERNAL/train_users_2.csv')


# train = pd.concat((train, test), axis = 0, ignore_index=True)


def values_to_categories(table):

    """
    Let's convert all of the categorical variables to a pandas category type.
    This will help with cleanliness, speed, and overall transparency between python libraries.

    """
    categorical_features = [
        'affiliate_channel',
        'affiliate_provider',
        'country_destination',
        'first_affiliate_tracked',
        'first_browser',
        'first_device_type',
        'gender',
        'language',
        'signup_app',
        'signup_method'
    ]
    for categorical_feature in categorical_features:
        table[categorical_feature] = table[categorical_feature].astype('category')

    return table


def convert_to_datetime(table):

    """
    Convert the time strings to pandas datetimes and get rid of the old timestamp column.
    :param table:
    :return: table
    """

    table['date_account_created'] = pd.to_datetime(table['date_account_created'])
    table['date_first_booking'] = pd.to_datetime(table['date_first_booking'])
    table['date_first_active'] = pd.to_datetime((table.timestamp_first_active // 1000000), format='%Y%m%d')
    table.drop('timestamp_first_active', axis=1, inplace=True)

    return table


def remove_outliers(df, column, min_val, max_val):
    """

    :param df:
    :param column:
    :param min_val:
    :param max_val:
    :return: df
    """

    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values <= min_val, col_values >= max_val), np.NaN, col_values)
    return df


def convert_to_binary(df, column_to_convert):
    """

    :param df:
    :param column_to_convert:
    :return: df
    """

    categories = list(df[column_to_convert].drop_duplicates())

    for category in categories:
        cat_name = str(category).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "").lower()
        col_name = column_to_convert[:5] + '_' + cat_name[:10]
        df[col_name] = 0
        df.loc[(df[column_to_convert] == category), col_name] = 1

    return df

# Arguably, the most important column in the dataset is the one the model will try to predict – country_destination.
# Looking at the number of records that fall into each category can help provide some insights into how the model
# should be constructed as well as pitfalls to avoid.


# Fixing age column
print("Fixing age column...")
df = remove_outliers(df = train, column='age', min_val=15, max_val=90)
df.age.fillna(-1, inplace=True)
print("Fixing first affiliate tracked column...")
df.first_affiliate_tracked.fillna(-1, inplace=True)


# print('Converting values to categories...')
# train = values_to_categories(train)

print('Converting times to datetimes...')
train = convert_to_datetime(train)


# One Hot Encoding
print("One Hot Encoding categorical data...")
columns_to_convert = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                      'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']

for column in columns_to_convert:
    train = convert_to_binary(df=train, column_to_convert=column)
    train.drop(column, axis=1, inplace=True)


# Add new date related fields
print("Adding new fields...")
train['day_account_created'] = train['date_account_created'].dt.weekday
train['month_account_created'] = train['date_account_created'].dt.month
train['quarter_account_created'] = train['date_account_created'].dt.quarter
train['year_account_created'] = train['date_account_created'].dt.year
train['hour_first_active'] = train['date_first_active'].dt.hour
train['day_first_active'] = train['date_first_active'].dt.weekday
train['month_first_active'] = train['date_first_active'].dt.month
train['quarter_first_active'] = train['date_first_active'].dt.quarter
train['year_first_active'] = train['date_first_active'].dt.year
train['created_less_active'] = (train['date_account_created'] - train['date_first_active']).dt.days


# Drop unnecessary columns
print('Dropping unnecessary date related columns..')
columns_to_drop = ['date_account_created', 'date_first_active', 'date_first_booking']
for column in columns_to_drop:
    if column in train.columns:
        train.drop(column, axis=1, inplace=True)

len(train)


with open('users.pickle', 'wb') as handle:
    pickle.dump(train, handle)

