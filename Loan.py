import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics

LoanDF = pandas.read_csv('Loan_Rejected.csv')
# XDF = LoanDF['Amount Requested']
YDF = LoanDF['Risk_Score']
# 轉成float, 移除字串中的%
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'].str.strip('%').astype('float32')
# 將數字轉化為百分比型式
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'] / 100
# print(LoanDF.head(1))

# todo: 處理其他特徵值
# https://www.kaggle.com/shikhar96/titanic-prediction-using-scikit-learn-tutorial

# todo: improve regression model
# https://stackoverflow.com/questions/47577168/how-can-i-increase-the-accuracy-of-my-linear-regression-modelmachine-learning

# 這資料格式不是對的
# trainX = pandas.DataFrame([XDF[:10000]], LoanDF['Debt-To-Income Ratio'][:10000]).T

# 應該也不是對的格式
# trainX = pandas.DataFrame([XDF[:10000]], LoanDF['Debt-To-Income Ratio'][:10000])

# trainX = pandas.DataFrame()
# trainX['Amount Requested'] = LoanDF['Amount Requested']
# trainX['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio']
# same as
X = LoanDF[['Amount Requested', 'Debt-To-Income Ratio']]

# print(X.info())
# print(X.head(5))

# 將資料分成訓練組及測試組
X_train, X_test, y_train, y_test = train_test_split(X, YDF, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

# 印出係數
print(lm.coef_)

# 印出截距
print(lm.intercept_)
print('Score of the multi linear regression:')
print(lm.score(X_test, y_test))
print(f_regression(X_test, y_test)[1])

# print(f_regression(trainX[:10000], YDF[:10000])[1][:1])

# 載入迴歸常見的評估指標
predictions = lm.predict(X_test)
# Mean Absolute Error (MAE)代表平均誤差，公式為所有實際值及預測值相減的絕對值平均。
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

# Mean Squared Error (MSE)比起MSE可以拉開誤差差距，算是蠻常用的指標，公式所有實際值及預測值相減的平方的平均
print('MSE:', metrics.mean_squared_error(y_test, predictions))

# Root Mean Squared Error (RMSE)代表MSE的平方根。比起MSE更為常用，因為更容易解釋y。
print('RMSE:', numpy.sqrt(metrics.mean_squared_error(y_test, predictions)))
