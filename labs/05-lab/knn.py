def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)
class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        #not sure what a distance matrix is
        #looked it up and it seems like the distance between pairs and stuff
        #not sure how to use that in this situation
        self.distance_matrix = ...

    def predict(self, example):
        #not sure what example is supposed to be, nor what to really do here
        return ...

    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", self.get_error(predicted, actual))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Add the dataset here

iris = datasets.load_iris()

x, y = iris.data, iris.target

# Split the data 70:30 and predict.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

# create a new object of class KNN

knn_object = KNN(10, x_train, y_train)

# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

species_list = []

for species_index in iris.target:
    species_list.append(iris.target_names[species_index])

df['Specie'] = species_list

df_setosa = df.loc[df['Specie'] == 'setosa']
df_setosa_sepal_length = df_setosa['sepal length (cm)']
df_setosa_sepal_width = df_setosa['sepal width (cm)']
df_setosa_petal_length = df_setosa['petal length (cm)']
df_setosa_petal_width = df_setosa['petal width (cm)']

df_versicolor = df.loc[df['Specie'] == 'versicolor']
df_versicolor_sepal_length = df_versicolor['sepal length (cm)']
df_versicolor_sepal_width = df_versicolor['sepal width (cm)']
df_versicolor_petal_length = df_versicolor['petal length (cm)']
df_versicolor_petal_width = df_versicolor['petal width (cm)']

df_virginica = df.loc[df['Specie'] == 'virginica']
df_virginica_sepal_length = df_virginica['sepal length (cm)']
df_virginica_sepal_width = df_virginica['sepal width (cm)']
df_virginica_petal_length = df_virginica['petal length (cm)']
df_virginica_petal_width = df_virginica['petal width (cm)']

plt.boxplot([df_setosa_sepal_length, df_versicolor_sepal_length, df_virginica_sepal_length], labels=['setosa', 'versicolor', 'virginica'])
plt.ylabel("Sepal length (cm)")
plt.savefig("./labs/05-lab/sepal-length-boxplot.png")
plt.close()

plt.boxplot([df_setosa_sepal_width, df_versicolor_sepal_width, df_virginica_sepal_width], labels=['setosa', 'versicolor', 'virginica'])
plt.ylabel("Sepal width (cm)")
plt.savefig("./labs/05-lab/sepal-width-boxplot.png")
plt.close()

plt.boxplot([df_setosa_petal_length, df_versicolor_petal_length, df_virginica_petal_length], labels=['setosa', 'versicolor', 'virginica'])
plt.ylabel("Petal length (cm)")
plt.savefig("./labs/05-lab/petal-length-boxplot.png")
plt.close()

plt.boxplot([df_setosa_petal_width, df_versicolor_petal_width, df_virginica_petal_width], labels=['setosa', 'versicolor', 'virginica'])
plt.ylabel("Petal width (cm)")
plt.savefig("./labs/05-lab/petal-width-boxplot.png")
plt.close()

# predict the labels using KNN

# use the test function to compute the error


# I just started following this https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75 

iris = datasets.load_iris()

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors= 10)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)

print(score)
#score is how accurate the predictions were when ran on x_test compared to real results y_test

