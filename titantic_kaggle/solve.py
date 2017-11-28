import pandas as pd
from sklearn import tree

frame = pd.read_csv('./train.csv')

clf = tree.DecisionTreeClassifier()


survive = frame['Survived']

# First num is rows 2nd is col index
data = frame.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

clf = clf.fit(data, survive)
