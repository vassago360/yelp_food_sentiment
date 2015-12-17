import os, csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random

def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x

#Go to the directory
os.chdir('C:\\Users\\user\\Documents\\CMPS242\\project')

#Get the CSVs and generate traindata* and traintarget arrays
devData = []  
devTarget = 0
featureIndexValsToNamesDict = dict()
        
reader = csv.reader(open('development_set_all_instances.csv', 'r'))
for i, row in enumerate(reader):
    if i == 0:
        indexVal = 0
        for j, item in enumerate(row):
            if 5 <= j <= 22:
                featureIndexValsToNamesDict[indexVal] = item
                indexVal += 1
            if 27 <= j <= 90:
                featureIndexValsToNamesDict[indexVal] = item
                indexVal += 1
    if i != 0:
        devDataRow = []
        for j, item in enumerate(row):
            if j == 4: #annot_mturk_mean_response_agreement
                devTarget = float(item)
            if 5 <= j <= 22: #turk worker features  5 <= j <= 22
                try:
                    devDataRow.append(float(item))
                except:
                    if item == "FALSE":
                        #trainRow.append("FALSE")
                        devDataRow.append(int(0))
                    if item == "TRUE":
                        #trainRow.append("TRUE")
                        devDataRow.append(int(1))
                    if item == "":
                        devDataRow.append(0)
            if 27 <= j <= 90: #computer generated features 27 <= j <= 90
                try:
                    devDataRow.append(float(item))
                except:
                    if item == "FALSE":
                        #trainRow.append("FALSE")
                        devDataRow.append(int(0))
                    if item == "TRUE":
                        #trainRow.append("TRUE")
                        devDataRow.append(int(1))
                    if item == "":
                        devDataRow.append(0)
        devDataRow.append(devTarget)
        devData.append(devDataRow)


trainData = np.array(devData)[:,:-1]
trainTarget =  np.array(devData)[:,-1:]
print("trainData shape before: " + str(trainData.shape))

kVal = 10
kBest = SelectKBest(f_regression, k=kVal).fit(trainData, trainTarget)
bestFeaturesIndex = kBest.get_support(indices=True)
trainDataKBestFeatures = kBest.transform(trainData)
print("trainData shape after: " + str(trainDataKBestFeatures.shape))
print("trainTarget shape: " + str(trainTarget.shape))

"""
#plot
plt.figure()
for i in range(kVal):
    r = lambda: random.randint(0,255)
    colorVal = ('#%02X%02X%02X' % (r(),r(),r()))
    if not (i in bestFeaturesIndex):
        plt.plot(Xintervals, Yintervals[i], c=colorVal, linewidth=0.5)
for i, indexVal in enumerate(bestFeaturesIndex):
    r = lambda: random.randint(0,255)
    colorVal = ('#%02X%02X%02X' % (r(),r(),r()))
    labelVal = "#" + str(i + 1) + " " + featureIndexValsToNamesDict[indexVal]
    plt.plot(Xintervals, Yintervals[indexVal], c=colorVal, label=labelVal, linewidth=0.5)
plt.xlabel("Examples trained on")
plt.ylabel("Mean Absolute Error")
plt.title("1-Feature Decision Tree Regression")
plt.legend(loc="best", fontsize="11")
plt.show()

featureIndexValsToNamesDict[bestIndices[i]]
"""
