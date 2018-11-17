import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

LoanDF = pandas.read_csv('Loan_Rejected.csv')
# XDF = LoanDF['Amount Requested']
YDF = LoanDF['Risk_Score']
# 轉成float, 移除字串中的%
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'].str.strip('%').astype('float32')
# 將數字轉化為百分比型式
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'] / 100
LoanDF = LoanDF.drop(['Employment Length', 'Risk_Score'], axis=1)


# 處理其他特徵值, R2來到0.092
# https://www.kaggle.com/shikhar96/titanic-prediction-using-scikit-learn-tutorial


def encode_features(df):
    features = ['Application Date', 'Loan Title', 'State']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


# todo: improve regression model
# https://stackoverflow.com/questions/47577168/how-can-i-increase-the-accuracy-of-my-linear-regression-modelmachine-learning
# X_LoanDF = encode_features(LoanDF)
X = LoanDF[['Amount Requested', 'Debt-To-Income Ratio']]

# 轉成np array, 可以加速
X = X.values

# 做curve fitting
# poly2 = PolynomialFeatures(degree=3)
# X_poly2 = poly2.fit_transform(X_LoanDF)

# 將資料分成訓練組及測試組
X_train, X_test, y_train, y_test = train_test_split(X, YDF, test_size=0.4, random_state=101)

# def train_and_evaluate(clf, X_train, y_train):
#     # 做cross validation
#     clf.fit(X_train, y_train)
#
#     print("Coefficient of determination on training set:", clf.score(X_train, y_train))
#
#     # create a k-fold cross validation iterator of k=5 folds
#     cv = KFold(n_splits=10, random_state=33)
#     scores = cross_val_score(clf, X_train, y_train, cv=cv)
#     print("Average coefficient of determination using 10-fold cross-validation:", numpy.mean(scores))


# train_and_evaluate(lm, X_train, y_train)

# 改random forest試試
rf = RandomForestRegressor(n_estimators=100, random_state=0, max_depth=16)
rf.fit(X_train, y_train)

# print(rf.predict([[1, 1]]))

# 載入迴歸常見的評估指標
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(numpy.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
print(mape)
# Calculate and display accuracy
accuracy = 100 - numpy.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
print(rf.score(X_test, y_test))
