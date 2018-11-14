from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

LoanDF = pandas.read_csv('../Loan_Rejected.csv')
# 轉成float, 移除字串中的%
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'].str.strip('%').astype('float32')
# 將數字轉化為百分比型式
LoanDF['Debt-To-Income Ratio'] = LoanDF['Debt-To-Income Ratio'] / 100

X = LoanDF[['Amount Requested', 'Debt-To-Income Ratio']]
Y = LoanDF['Risk_Score']


# todo: deep learning要將dataframe轉成ndarray

# todo: 之後還要標準化

def baseline_model():
    # create model
    model = Sequential()
    # input layer
    model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
    # output layer
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# The Keras wrapper object for use in scikit-learn as a regression estimator is called KerasRegressor.
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=500, verbose=2)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
