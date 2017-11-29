import pandas as pd
from sklearn import tree, preprocessing
from sklearn.ensemble import RandomForestClassifier

train_frame = pd.read_csv('./train.csv')
test_frame = pd.read_csv('./test.csv')
survive = train_frame['Survived']

used_fields = ['Sex']
final_fields = ['PassengerId', 'Survived']


def clean_data(data):
    d = train_frame.loc[:, used_fields]
    d = d.fillna(0)
    le = preprocessing.LabelEncoder()
    le.fit(['male', 'female'])
    transform_sex = le.transform(d['Sex'])
    d['Sex'] = transform_sex
    return d


def classify(sample, test, clf):
    clf.fit(sample, survive)
    return clf.predict(test)


def solve(fn, clf):
    train = clean_data(train_frame)
    test = clean_data(test_frame)
    r = classify(train, test, clf)
    original_test = pd.read_csv('./test.csv')
    original_test['Survived'] = pd.Series(r)
    original_test = original_test.loc[:, final_fields]
    original_test.to_csv('{}.csv'.format(fn), index=False)


solve('forest', RandomForestClassifier(verbose=True))
solve('tree', tree.DecisionTreeClassifier())



