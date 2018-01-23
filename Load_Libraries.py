# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
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
def head():
    print(dataset.head(20))


# descriptions
def descriptions():
    print(dataset.describe())


# class distribution
def class_distribution():
    print(dataset.groupby('class').size())


# box and whisker plots
def box_and_whisker_plots():
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()


# histograms
def histograms():
    dataset.hist()
    plt.show()


# scatter plot matrix
def scatter_plot_matrix():
    scatter_matrix(dataset)
    plt.show()

box_and_whisker_plots()
histograms()
scatter_plot_matrix()
