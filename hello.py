from sklearn import tree

# "Hello world" of ML

features = [[140, 1], [130, 1], [150, 2], [170, 2]]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[130, 1]]))

