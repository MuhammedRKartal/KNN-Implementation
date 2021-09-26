#Author: Muhammed Rahmetullah Kartal

# The KNN (K-Nearest Neighbors) Classifier algorithm
# selects a test input and tries to classify it related to train data,
# this prediction is done in several steps,
# at first it calculates the distance between test input and each member of train data one by one,
# in this project these calculations are done by the Euclidian Distance method.
# After that calculations it sorts the data and selects the smallest k neighbors,
# this k value is decided by the user, after that selection related to user’s decision
# it selects the best one of them, in this project that decision is done with the inverse-distance weights method,
# this method decides the best neighbor by checking its distance to each train data element,
# let’s assume the distance to first element of train data is d0,
# and first element’s class is ‘A’ algorithm will add 1/d0 to ‘A’ class counter
# and continues that operation with each one of the selected K values.

from math import sqrt
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#calculates accuracy and error rate in a dataframe
def calculate_accuracy(df):
    counter = 0
    for i in range(df.shape[0]):
        if df[0].loc[i] != df['predicted'].loc[i]:
            counter += 1
    errCount = str(counter) + '/' + str(df.shape[0])
    return (1 - (counter / df.shape[0])), errCount

#weights metric of knn
def bestK(differentKs):
    oneCount = 0 #count of class 1
    twoCount = 0 #count of class 2
    threeCount = 0 #count of class 3
    for i in range(len(differentKs)):
        choice = differentKs[i] #choosen element of shortest distanced K values
        distance = choice[0] #taking the distance value of choosen element
        element = data[0].loc[choice[1]] #finding the element in original train data

        #if distance is 0, to avoid division by 0 error value is setted to 10^-7
        if distance == 0:
            distance = (1 / 10000000)

        #if element is 1, add 1/distance to count of class 1
        if element == 1:
            oneCount += (1 / distance)
        # if element is 2, add 1/distance to count of class 2
        elif element == 2:
            twoCount += (1 / distance)
        # if element is 3, add 1/distance to count of class 3
        elif element == 3:
            threeCount += (1 / distance)

    #return the biggest count's value
    if oneCount > twoCount and oneCount > threeCount:
        return 1
    elif twoCount > oneCount and twoCount > threeCount:
        return 2
    elif threeCount > oneCount and threeCount > twoCount:
        return 3

#K-Nearest Neighbours Algorithm, returns a dataframe
def KNN(data, test, k):
    distancesList = list()  # list of distances of all different test elements
    for j in range(test.shape[0]):
        distances = list()  # list of distances of one test element
        for i in range(data.shape[0]):

            #Euclidean Distance calculation
            x = (data[1].loc[i] - test[1].loc[j]) ** 2
            y = (data[2].loc[i] - test[2].loc[j]) ** 2
            distance = sqrt(x + y)

            distances.append([distance, i]) #the values is going to be sorted so i is going
            # to be used to find element in original array
        distancesList.append(distances)
        distancesList[j] = sorted(distancesList[j]) #sorting the values related to distances
        differentKs = distancesList[j][:k] #taking first K of these distances
        test['predicted'].loc[j] = bestK(differentKs) #selecting the best K value
    return test

#Same algorithm to draw decision boundaries (returns numpy array instead of dataframe)
def KNNArray(data, test, k):
    distancesList = list()  # list of distances of all different test users
    predictionArray = np.zeros(len(test))
    for j in range(len(test)):
        distances = list()  # list of distances of one test user
        for i in range(data.shape[0]):
            x = (data[1].loc[i] - test[j][0]) ** 2
            y = (data[2].loc[i] - test[j][1]) ** 2
            distance = sqrt(x + y)

            distances.append([distance, i])
        distancesList.append(distances)
        distancesList[j] = sorted(distancesList[j])
        differentKs = distancesList[j][:k]
        predictionArray[j] = bestK(differentKs)
    return predictionArray

def calcAccuracySK(data,test,i):
    test["predicted"] = None #creating a predicted values column on test dataframe

    #inverse-weighted Knn algorithm by using euclidean distance calculation metric with different neighbour numbers
    knn = KNeighborsClassifier(n_neighbors=i, metric='euclidean', weights='distance')

    y = data[0] #correct values of predictions
    x = data.drop([0], axis=1) #features
    testIn = test.drop([0,'predicted'], axis=1) #test input data with just x and y coordinate features
    knn.fit(x, y) #fitting the data
    test['predicted'] = knn.predict(testIn) #predicting

    #calculation of accuracy and error count
    sklearn_accuracy = calculate_accuracy(test)
    sklearn_errorcount = sklearn_accuracy[1]
    sklearn_accuracy = sklearn_accuracy[0]

    return (i, sklearn_accuracy, sklearn_errorcount)


def calcAccuracy(data,test,i):
    test["predicted"] = None
    test = KNN(data, test, i)

    #calculation of accuracy and error count
    my_accuracy = calculate_accuracy(test)
    my_errorcount = my_accuracy[1]
    my_accuracy = my_accuracy[0]

    return (i, my_accuracy, my_errorcount)


def drawGraph(data, test, k):
    test['predicted'] = None #adding predictions column to test data's dataframe
    test = KNN(data, test, k) #running the algorithm

    y = test['predicted'] #seperating the predictions
    X = test.drop([0, 'predicted'], axis=1) #seperating the features

    #
    h = .02  # step size in the mesh
    X = X.to_numpy()

    #finding min and max values for coordinates
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #continious values to create boundaries
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = KNNArray(data, (np.c_[xx.ravel(), yy.ravel()]), k)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn) #plotting decision boundaries with matplotlib
    scp = sns.scatterplot(data=test, x=test[1], y=test[2], hue=test['predicted']) #seaborn scatter plot
    scp.set(xlabel='X Coordinates', ylabel = 'Y Coordinates',title='KNN Decisions') #seaborn scattter plot axis naming

    # plt.scatter(test[1], test[2], c=test['predicted'])  #scatter plot with matplotlib (alternative solution)

    plt.show()


accuracies = []
accuraciesSK = []


#Reading test and train datas
data = pd.read_csv("data_training.csv", header=None) #Train data
test = pd.read_csv("data_test.csv", header=None) #Test data


# Printing the comparisons
for i in [1, 3, 5, 7, 9, 11, 13, 15]:
    accuracies.append(calcAccuracy(data,test,i))
    accuraciesSK.append(calcAccuracySK(data,test,i))

print("My KNN")
print(tabulate(accuracies, headers=['K', 'Accuracy', 'Error Count']))

print("\nSklearn KNN")
print(tabulate(accuraciesSK, headers=['K', 'Accuracy', 'Error Count']))

# Printing the graph
drawGraph(data,test,25)


# plotting train data
trainscatter = sns.scatterplot(data=data, x=data[1], y=data[2], hue = data[0])
trainscatter.set(xlabel='X Coordinates', ylabel = 'Y Coordinates',title='Train Data')
plt.show()