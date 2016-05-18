# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, tree
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
# from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import operator
import pickle
import pandas as pd


path = '../../'

df_all = pickle.load(open(path + "final.pickle", "rb"))

# create label column and drop useless data from data set
print('Preparing data...')
y = df_all.country_destination
df_all.drop(['country_destination'], axis=1, inplace=True)
# df_all.drop(['id'], axis=1, inplace=True)
X = df_all

# Encode labels
print('Encoding labels...')
labels = y.values
le = preprocessing.LabelEncoder()
y = le.fit_transform(labels)

# split data into training and test sets
print('splitting data into train, validation, and test..')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.20)

# split training into training and validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.20)

# Classifier
print('Running gradient boosting on training data...')
params = {'eta': 0.2,
          'max_depth': 6,
          'subsample': 0.5,
          'colsample_bytree': 0.5,
          'objective': 'multi:softprob',
          'num_class': 13}
num_boost_round = 1
d_train = xgb.DMatrix(x_train, y_train)
clf1 = xgb.train(params=params, dtrain=d_train, num_boost_round=num_boost_round)

# Get feature scores and store in DataFrame
print('Gathering feature scores into data frame')
importance = clf1.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)),
    columns=['feature', 'fscore']
)
print('---------------------------------------------------------------')
print('Importance of features based on gradient boosting score')
print(importance_df.tail(30))

# Get prediction values and residuals on validation set
print('getting predictions')
y_preds = clf1.predict(xgb.DMatrix(x_valid))
