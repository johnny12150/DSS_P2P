import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
import numpy
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# the dataset is too larage please download from 'https://www.kaggle.com/wendykan/lending-club-loan-data'
DF = pandas.read_csv('../loan.csv')
# drop entire null columns
DF = DF.dropna(axis=1, how='all')
DF['emp_length'] = DF.emp_length.astype(str)
# DF['term'] = DF.term.astype(str)
# DF['purpose'] = DF.purpose.astype(str)
# DF['loan_status'] = DF.loan_status.astype(str)
# DF['grade'] = DF.grade.astype(str)
# DF['verification_status'] = DF.verification_status.astype(str)
# DF['home_ownership'] = DF.home_ownership.astype(str)


# Select the columns we need
column_needed = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'term',
                 'home_ownership', 'loan_status', 'purpose', 'verification_status', 'addr_state', 'zip_code',
                 'emp_length', 'annual_inc']

LoanDF = DF[column_needed].copy()

#  只取兩種grade的row
# LoanDF = LoanDF.loc[LoanDF['grade'].isin(['A', 'B'])]
# LoanDF = LoanDF[(LoanDF.grade == 'A') | (LoanDF.grade == 'B')]

# 將用不到的欄位在這裡先drop
# Y = LoanDF[['sub_grade']]
Y = LoanDF[['grade']]
LoanDF = LoanDF.drop('grade', axis=1)
LoanDF = LoanDF.drop('sub_grade', axis=1)
LoanDF = LoanDF.drop('int_rate', axis=1)
LoanDF = LoanDF.drop('id', axis=1)
LoanDF = LoanDF.drop('member_id', axis=1)
LoanDF = LoanDF.drop('funded_amnt', axis=1)
LoanDF = LoanDF.drop('addr_state', axis=1)
LoanDF = LoanDF.drop('zip_code', axis=1)
LoanDF = LoanDF.drop('installment', axis=1)
print(LoanDF.shape)
print(Y.shape)


# LoanDF = LoanDF.drop('loan_status', axis=1)
# LoanDF = LoanDF.drop('verification_status', axis=1)
# LoanDF = LoanDF.drop('term', axis=1)
# LoanDF = LoanDF.drop('purpose', axis=1)
# LoanDF = LoanDF.drop('emp_length', axis=1)


def encode_features(df):
    features = ['term', 'home_ownership', 'loan_status', 'purpose', 'verification_status', 'emp_length']
    # features = ['home_ownership']
    for feature in features:
        # df[feature] = df[feature].fillna('0')
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        # save each encoder with feature name
        joblib.dump(le, 'encoder/le_%s.pkl' % feature)
        one_hot = preprocessing.OneHotEncoder()
        ohe = one_hot.fit(df[[feature]])
        feature_array = ohe.transform(df[[feature]]).toarray()
        # save each encoder with feature name
        joblib.dump(ohe, 'encoder/ohe_%s.pkl' % feature)
        feature_labels = list(le.classes_)
        one_hot_features = pandas.DataFrame(feature_array, columns=feature_labels)
        df = pandas.concat([df, one_hot_features], axis=1)
        # df = pandas.concat([df, one_hot_features], axis=1, join='inner', join_axes=[df.id])
        # todo: df shape 改變了 (因為fillna)
        # drop舊的(concat前的, ex: term...)
        df = df.drop(feature, axis=1)
    return df


# feature
X = encode_features(LoanDF)


# LoanDF = LoanDF.drop('id', axis=1)

def Y_encode_features(df):
    # features = ['sub_grade']
    features = ['grade']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])  # Transform Categories Into Integers
        joblib.dump(le, 'encoder/Yle_%s.pkl' % feature)
        one_hot = preprocessing.OneHotEncoder()
        ohe = one_hot.fit(df[[feature]])
        feature_array = ohe.transform(df[[feature]]).toarray()
        joblib.dump(ohe, 'encoder/Yohe_%s.pkl' % feature)
        feature_labels = list(le.classes_)
        one_hot_features = pandas.DataFrame(feature_array, columns=feature_labels)
        df = pandas.concat([df, one_hot_features], axis=1)
        df = df.drop(feature, axis=1)
    return df


Y = Y_encode_features(Y)

X = X.values
Y = Y.values

# 標準化
minMaxScale = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaler = minMaxScale.fit(X)

# scaler = StandardScaler().fit(X)
# save minMaxScaler
joblib.dump(scaler, "scaler/XminMAX.pkl")

# joblib.dump(scaler, "scaler/Xstandard.pkl")

X = scaler.transform(X)

# 將資料分成訓練組及測試組
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# 建立模型
model = Sequential()
model.add(Dense(36, input_dim=49, activation='relu'))
model.add(Dense(30, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
# # model.add(Dense(35, kernel_initializer='normal', activation='softmax'))
model.add(Dense(7, kernel_initializer='normal', activation='softmax'))
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
train_history = model.fit(x=X_train, y=y_train, validation_split=0.1, epochs=10, batch_size=5000, verbose=2)

model.save('model/MLP_model.h5')

scores = model.evaluate(x=X_test, y=y_test, verbose=0)
print(scores[1])
