from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
print("iris datasets loaded")
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)
print("dataset is training and testing ....")
print("size of training data and its label", x_train.shape, y_train.shape)
print("size of training data and its label", x_test.shape, y_test.shape)
for i in range(len(iris.target_names)):
        print("label", i, "-", str(iris.target_names[i]))
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
print("Results of classification using K-NN K=1")
for r in range(0, len(x_test)):
        print("Sample: ", str(x_test[r]), "Actual-Label: ", str(y_test[r]), "Predicted-Label: ", str(y_pred[r]))
print("Classification Accuracy: ", classifier.score(x_test, y_test))
