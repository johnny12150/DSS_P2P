import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
import numpy
import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

# the dataset is too larage please download from 'https://www.kaggle.com/wendykan/lending-club-loan-data'
DF = pandas.read_csv('../loan.csv')
# drop entire null columns
DF = DF.dropna(axis=1, how='all')
DF['emp_length'] = DF.emp_length.astype(str)

# Select the columns we need
column_needed = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'term',
                 'home_ownership', 'loan_status', 'purpose', 'verification_status', 'addr_state', 'zip_code',
                 'emp_length', 'annual_inc']

LoanDF = DF[column_needed].copy()

# 將用不到的欄位在這裡先drop
# Y = LoanDF[['sub_grade']]
Y = LoanDF[['grade']]
LoanDF = LoanDF.drop('grade', axis=1)
LoanDF = LoanDF.drop('sub_grade', axis=1)
# LoanDF = LoanDF.drop('int_rate', axis=1)
LoanDF = LoanDF.drop('id', axis=1)
LoanDF = LoanDF.drop('member_id', axis=1)
LoanDF = LoanDF.drop('funded_amnt', axis=1)
LoanDF = LoanDF.drop('addr_state', axis=1)
LoanDF = LoanDF.drop('zip_code', axis=1)
LoanDF = LoanDF.drop('installment', axis=1)


def encode_features(df):
    features = ['term', 'home_ownership', 'loan_status', 'purpose', 'verification_status', 'emp_length']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        # save each encoder with feature name
        # joblib.dump(le, 'encoder/le_%s.pkl' % feature)
        one_hot = preprocessing.OneHotEncoder()
        ohe = one_hot.fit(df[[feature]])
        feature_array = ohe.transform(df[[feature]]).toarray()
        # save each encoder with feature name
        # joblib.dump(ohe, 'encoder/ohe_%s.pkl' % feature)
        feature_labels = list(le.classes_)
        one_hot_features = pandas.DataFrame(feature_array, columns=feature_labels)
        df = pandas.concat([df, one_hot_features], axis=1)
        # drop舊的(concat前的, ex: term...)
        df = df.drop(feature, axis=1)
    return df


# feature
X = encode_features(LoanDF)


def Y_encode_features(df):
    # features = ['sub_grade']
    features = ['grade']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])  # Transform Categories Into Integers
        # joblib.dump(le, 'encoder/Yle_%s.pkl' % feature)
        # one_hot = preprocessing.OneHotEncoder()
        # ohe = one_hot.fit(df[[feature]])
        # feature_array = ohe.transform(df[[feature]]).toarray()
        # # joblib.dump(ohe, 'encoder/Yohe_%s.pkl' % feature)
        # feature_labels = list(le.classes_)
        # one_hot_features = pandas.DataFrame(feature_array, columns=feature_labels)
        # df = pandas.concat([df, one_hot_features], axis=1)
        # df = df.drop(feature, axis=1)
    return df


Y = Y_encode_features(Y)

X = X.values
# 避免從df轉nd_array的過程中有產生inf/ nan值
X = numpy.nan_to_num(X)
Y = Y.values

# 標準化
# minMaxScale = preprocessing.MinMaxScaler(feature_range=(0, 1))
# scaler = minMaxScale.fit(X)

scaler = StandardScaler().fit(X)
# save minMaxScaler
# joblib.dump(scaler, "scaler/XminMAX.pkl")

# joblib.dump(scaler, "scaler/Xstandard.pkl")

X = scaler.transform(X)

# 將資料分成訓練組及測試組
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# SVM跟Logistic Regression一樣，Y只支援一維的input
y_train = y_train.reshape(-1)

# 建立模型
Clf = KNeighborsClassifier()
Clf.fit(X_train, y_train)

scores = Clf.score(X_test, y_test)
print(scores)
