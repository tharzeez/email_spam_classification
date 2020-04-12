import numpy as np
import pandas
import nltk
import matplotlib.pyplot as plt
import sys

# xValue is the keywords that define whether or not an email is spam
xValue = [
    "software",
    "count",
    "mmbtu",
    "attached",
    "receive",
    "Important information regarding",
    "Additional income",
    "#1",
    "100% more",
    "100% free",
    "100 %",
    "forwarded by",
    "100% satisfied",
    "Best price",
    "drugs",
    "enron",
    "pills",
    " hot ",
    "Billion",
    "Cash bonus",
    "Double your income",
    "Earn extra cash",
    "Earn money",
    "Extra cash",
    "information",
    "Free access",
    "Free consultation",
    "Free gift",
    "texas",
    "know",
    "Free hosting",
    "Free money",
    "Free quote",
    "Free trial",
    "Full refund",
    "Get paid",
    "Giveaway",
    "Guaranteed",
    "Lowest price",
    "Make money",
    "Million dollars",
    "Miracle",
    "Money back",
    "Once in a lifetime",
    "One time",
    "Pennies a day",
    "Potential earnings",
    "Prize",
    "Promise",
    "Pure profit",
    "Risk-free",
    "Satisfaction guaranteed",
    "Save big money",
    "Save up to",
    "Special promotion",
    "Act now",
    "Apply now",
    "Become a member",
    "Call now",
    "Click below",
    "Click here",
    "Get it now",
    "Do it today",
    "Don’t delete",
    "Information you requested",
    "Instant",
    "Limited time",
    "Order now",
    "Please read",
    "See for yourself",
    "Sign up free",
    "Take action",
    "This won’t last",
    "Urgent",
    "What are you waiting for",
    "While supplies last",
    "Winner",
    "Winning",
    "You are a winner",
    "You have been selected",
    "Bulk email",
    "Buy direct",
    "Cancel at any time",
    "Check or money order",
    "Congratulations",
    "Confidentiality",
    "Cures",
    "Penis",
    "Direct email",
    "Direct marketing",
    "Hidden charges",
    "Human growth hormone",
    "Internet marketing",
    "Lose weight",
    "Mass email",
    "No cost",
    "No fees",
    "enlargement",
    "No investment",
    "No questions asked",
    "Not junk",
    "Unsecured debt",
    "Unsolicited",
    "Valium",
    "Viagra",
    "Vicodin",
    "doctor",
    "Weight loss",
    "Xanax",
    "medication",
    "All new",
    "As seen on",
    "Billing",
    "Bonus",
    "Cash",
    "Certified",
    "Cheap",
    "please",
    "Claims",
    "Clearance",
    "Deal",
    "Debt",
    "Discount",
    "Fantastic",
    "Income",
    "Investment",
    "Join millions",
    "Lifetime",
    "Loans",
    "Luxury",
    "Message contains",
    "Name brand",
    "Offer",
    "meter",
    "Opt in",
    "Quote",
    "daren",
    "http",
    "please",
    "thanks",
    "farmer",
    "original",
    "robert",
    "see",
    "out",
    "volume",
    "my",
    "month",
    "contract",
    "corp",
    "energy",
    "statement",
    "email",
    "today",
    "file",
    "first",
    "following",
    "sitara",
    "hpl",
    "know",
    "gas",
    "development",
    "jackie",
    "Rates",
    "Refinance",
    "Removal",
    "report",
    "Score",
    "Subject to",
    "Terms and conditions",
    "Trial",
    "Unlimited",
    "Warranty",
    " 000 ",
    "business",
    "week",
]


featureSize = len(xValue) + 1 # featureSize + 1 for bias
theta1 = np.array([0 for x in range(featureSize)])

theta1 = theta1.reshape(featureSize, 1)


dataToRead = 3163
alpha = 0.1


df = pandas.read_excel('spam_ham_dataset.xlsx')
trainingData = np.array(df.iloc[:dataToRead, 2])
yTrainingData = np.array(df.iloc[:dataToRead, 3])

yTrainingData = yTrainingData.reshape(dataToRead,1)

# Creates a list containing 5 lists, each of 8 items, all set to 0
# w, h = 8, 5;
# Matrix = [[0 for x in range(w)] for y in range(h)] 

# features set ranging from 0:xValue for every single trainingData, i.e 3163 X thetaSize
# i.e X = 3163 X thetaSize
# create the feature set from the training data
# creates a list containing 3163 lists, each of thetaSize items, all set to 0
X = [[0 for x in range(featureSize - 1)] for y in range(dataToRead)]

X = np.array(X)

nltk.download('punkt')

count = 0
for dataIndex, data in enumerate(trainingData):
  for featureIndex, features in enumerate(xValue):
    if features.lower() in data.lower():
      count = count + 1
      X[dataIndex][featureIndex] = 1
    else:
      X[dataIndex][featureIndex] = 0

print(count)


X = np.insert(X, 0, 1, axis = 1) # add bias variable


def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))


# X is the featurization of 3163 X 1

# Hypothesis hTheta = sigmoid(theta * X)

max_iter = 15000 # change the iteration value
# max_iter = 2
cost = np.zeros((max_iter, 1))
for dummyCounter in range(max_iter):
  z = np.matmul(X, theta1)

  hypothesis = sigmoid(z) # 3163 X thetaSize


  # J(θ)= (−yTlog⁡(h)−(1−y)Tlog⁡(1−h))/m

  firstPart = np.matmul(np.transpose(yTrainingData), np.log(hypothesis))
  secondPart = np.matmul(np.transpose(np.subtract(1, yTrainingData)), np.log(np.subtract(1, hypothesis)))
  cost[dummyCounter] = np.divide(np.negative(np.add(firstPart, secondPart)), dataToRead)
  print(cost[dummyCounter].reshape(-1))


  gradient = np.matmul(np.transpose(X), np.subtract(hypothesis, yTrainingData))

  theta1 = np.subtract(theta1, np.divide(np.multiply(alpha, gradient), dataToRead))



testDataStarts = 3168 # simply hardcoded
testingData = np.array(df.iloc[testDataStarts:, 2])
yTestingData = np.array(df.iloc[testDataStarts:, 3])

totalTestData = 2006 # calculated manually
yTestingData = yTestingData.reshape(totalTestData,1)

XTestData = [[0 for xx in range(featureSize - 1)] for yx in range(totalTestData)]

XTestData = np.array(XTestData)


count = 0
for dataIndex, data in enumerate(testingData):
  for featureIndex, features in enumerate(xValue):
    # print(features)
    if features.lower() in data.lower():
      count = count + 1
      XTestData[dataIndex][featureIndex] = 1
    else:
      # print('not inside')
      XTestData[dataIndex][featureIndex] = 0

XTestData = np.insert(XTestData, 0, 1, axis = 1) # add bias variable


def f(x):
  return np.int(x)

f2 = np.vectorize(f)

for i in range(totalTestData):
  z = np.matmul(XTestData, theta1)
  hypothesis = sigmoid(z) > 0.5 # threshold value

hypothesis = f2(hypothesis)


correctClassification  = 0
wrongClassification = 0
for i in range(totalTestData):
  if int(hypothesis[i]) == int(yTestingData[i].reshape(-1)):
    correctClassification = correctClassification + 1
  else:
    wrongClassification = wrongClassification + 1

print(correctClassification, 'correct')
print(wrongClassification, 'wrong')
plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()