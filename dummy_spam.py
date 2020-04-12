import numpy as np
import pandas
import nltk
import matplotlib.pyplot as plt
import sys

# xValue is the keywords that define whether or not an email is spam
xValue = [
    "pornographic",
    "cheap",
]

# Theta will be of range 192 X 1

featureSize = len(xValue)
theta1 = np.array([0 for x in range(featureSize)])

theta1 = theta1.reshape(featureSize, 1)

dataToRead = 4

df = pandas.read_excel('spam_ham_dataset.xlsx')
trainingData = np.array(df.iloc[:dataToRead, 2])
yTrainingData = np.array(df.iloc[:dataToRead, 3])

yTrainingData = yTrainingData.reshape(dataToRead,1)

# Creates a list containing 5 lists, each of 8 items, all set to 0
# w, h = 8, 5;
# Matrix = [[0 for x in range(w)] for y in range(h)] 

# features set ranging from 0:xValue for every single trainingData, i.e 3163 X 192
# i.e X = 3163 X 192
# create the feature set from the training data
# creates a list containing 3163 lists, each of 192 items, all set to 0
X = [[0 for x in range(featureSize)] for y in range(dataToRead)]

X = np.array(X)
tokenizedTrainingData = []

nltk.download('punkt')

count = 0
for dataIndex, data in enumerate(trainingData):
  for featureIndex, features in enumerate(xValue):
    # print(features)
    if features.lower() in data.lower():
      count = count + 1
      X[dataIndex][featureIndex] = 1
    else:
      # print('not inside')
      X[dataIndex][featureIndex] = 0



def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))


# print(X)
print(len(X))

# X is the featurization of 3163 X 1

# Hypothesis hTheta = sigmoid(theta * X)

z = np.matmul(X, theta1)

hypothesis = sigmoid(z) # 3163 X 192


# J(θ)= (−yTlog⁡(h)−(1−y)Tlog⁡(1−h))/m

# ones = np.array([1 for yVal in range(len(yTrainingData))]) 



# ones = ones.reshape(dataToRead, 1) # 3163 X 1
# print(ones)
# print(ones.shape)

firstPart = np.matmul(np.transpose(yTrainingData), np.log(hypothesis))

# secondPart = np.multiply(np.subtract(ones, yNP), np.log(np.subtract(1, hypothesis)))
secondPart = np.matmul(np.transpose(np.subtract(1, yTrainingData)), np.log(np.subtract(1, hypothesis)))


costFunction = np.divide(np.negative(np.add(firstPart, secondPart)), dataToRead)
print(costFunction.reshape(-1))


gradient = np.matmul(np.transpose(X), np.subtract(hypothesis, yTrainingData))


print(gradient)

alpha = 0.001

theta1 = np.subtract(theta1, np.divide(np.multiply(alpha, gradient), dataToRead))

print(theta1)





