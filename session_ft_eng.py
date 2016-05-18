
import pandas as pd
import pickle
import numpy as np
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from sklearn.cross_validation import train_test_split
# from sklearn import preprocessing, tree
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
# from xgboost.sklearn import XGBClassifier


path = '../../'

print('Loading data...')
users = pickle.load( open( path + "users.pickle", "rb" ))
sessions = pd.read_csv( path + 'sessions.csv')


def find_unique(col, sessions_df, new_col_name):

    num_actions = pd.DataFrame(sessions_df.groupby(['user_id'])[col].nunique())
    num_actions.columns = [new_col_name]
    return num_actions.reset_index()


def fix_table(table):

    levels = table.columns.levels
    labels = table.columns.labels
    table.columns = levels[1][labels[1]]
    table.fillna(0, inplace=True)
    return table

# create separate tables for actions and device usage
print("creating actions data frames...")
actions = sessions.groupby(['user_id', 'action'], sort=False).sum().unstack(1)
print('creating devices data frames...')
devices = sessions.groupby(['user_id', 'device_type'], sort=False).sum().unstack(1)

# adjust data frames so that the indices are correct
print('fixing data frames...')
actions = fix_table(actions)
devices = fix_table(devices)

# create separate columns for number of actions and number of devices
print('Counting number of actions per user...')
num_actions = find_unique('action', sessions, 'num_actions')
num_actions.set_index('user_id', inplace=True)


print('Counting number of devices per user...')
num_devices = find_unique('device_type', sessions, 'num_devices')
num_devices.set_index('user_id', inplace=True)


# Merge device and actions data sets
print('Merging sessions dataframes...')
result_actions = pd.concat([actions, num_actions], axis=1, join="outer")
result_devices = pd.concat([devices, num_devices], axis=1, join="outer")
result = pd.concat([result_actions, result_devices], axis=1, join="outer")
print("Sessions length: " + str(result.shape[0]) + " sessions columns: " + str(result.shape[1]))
df_sessions = result.fillna(0)


# Merge user and session data sets
print('Merging users and sessions dataframes...')
users.set_index('id', inplace=True)
df_all = pd.concat([users, df_sessions], axis=1, join='inner')

# pickle data frame to use in modeling
print('Pickling dataframe...')
with open(path + 'final.pickle', 'wb') as handle:
    pickle.dump(df_all, handle)

print('HOORAY!')
