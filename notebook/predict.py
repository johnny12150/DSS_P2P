import pandas as pd
from sklearn.externals import joblib
from keras.models import load_model
import sys

# TEST: python predict.py 4000 " 36 months" RENT "Fully Paid" car Verified "5 years" 50000
input_list = [[]]

for index, arg in enumerate(sys.argv):
    if index != 0:
        input_list[0].append(arg)

print(input_list)
loaded_model = load_model('model/MLP_model.h5')


def feature_encode(df):
    features = ['term', 'home_ownership', 'loan_status', 'purpose', 'verification_status', 'emp_length']
    for feature in features:
        le = joblib.load('encoder/le_%s.pkl' % feature)
        df[feature] = le.transform(df[feature])
        ohe = joblib.load('encoder/ohe_%s.pkl' % feature)
        featured = ohe.transform(df[[feature]])
        feature_array = featured.toarray()
        feature_labels = list(le.classes_)
        one_hot_features = pd.DataFrame(feature_array, columns=feature_labels)
        df = pd.concat([df, one_hot_features], axis=1)
        df = df.drop(feature, axis=1)
    return df


# input_list = [[4000, ' 36 months', 'RENT', 'Fully Paid', 'car', 'Verified', '5 years', 50000]]
fake_new_input = pd.DataFrame(input_list, columns=['loan_amt', 'term', 'home_ownership', 'loan_status', 'purpose',
                                                   'verification_status', 'emp_length', 'annual_inc'])

predict_X = feature_encode(fake_new_input)
predict_X = predict_X.values
minMax = joblib.load("scaler/XminMAX.pkl")
predict_X = minMax.transform(predict_X)
predictions = loaded_model.predict_classes(predict_X)

print(predictions[0])
sys.stdout.flush()
