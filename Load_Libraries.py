# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# Shape
def shape():
    print(dataset.shape)


# Head
def head(i):
    print(dataset.head(i))


# descriptions
def descriptions():
    print(dataset.describe())


# class distribution
def class_distribution():
    print(dataset.groupby('class').size())


# box and whisker plots
def box_and_whisker_plots():
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()


# histograms
def histograms():
    dataset.hist()
    plt.show()


# scatter plot matrix
def scatter_plot_matrix():
    scatter_matrix(dataset)
    plt.show()


# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Let’s evaluate 6 different algorithms:

# Logistic Regression (LR)  simple linear
# Linear Discriminant Analysis (LDA)  simple linear
# K-Nearest Neighbors (KNN)  nonlinear
# Classification and Regression Trees (CART)  nonlinear
# Gaussian Naive Bayes (NB)  nonlinear
# Support Vector Machines (SVM).

# Spot Check Algorithms
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

# evaluate each model in turn
results = []
names = []


def evaluate_each_model_in_turn():
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# Compare Algorithms
def compare_algorithms():
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

