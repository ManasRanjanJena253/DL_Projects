# Importing dependencies

import numpy as np

# Creating a sigmoid perceptron class
print(np.exp(1))

class SigmoidPerceptron():
    def __init__(self,X_size):   # input_size is the no. of our inputs/no. of features, we are feeding to the perceptron.
        self.weights = np.random.randn(X_size)  # Generating an array having same no. of elements as the no. of inputs whose values are taken at random in the range of 0 to 1.
        self.bias = np.random.randn(1)   # Will generate an array of one element containing any random value between 0 and 1.

    def sigmoid(self,z):
        return 1/(1 * np.exp(-z))

    def predict(self,input_):
        z = np.dot(self.weights,input_) + self.bias
        return self.sigmoid(z)

    def fit(self,X,Y,learning_rate,num_epochs):
        num = X.shape[0]  # num is the no. of datapoints/features we have.

        # Here we have two for loops instead of one as in ml models because in the ml models we use simple gradient descent, and we need to update the weights after predicting the values for each datapoint present in X.
        # Whereas in the perceptron model we use stochastic gradient descent, and we update the data after prediction of each individual datapoint.
        for epoch in range(num_epochs):
            for k in range (num):
                input_ = X[k]
                target = Y[k]
                prediction = self.predict(input_)
                error = target-prediction

                # Update weights
                self.weights = self.weights +( learning_rate * error * prediction * (1 - prediction) * input_)

                # Update bias
                self.bias = self.bias + (learning_rate * error * prediction * (1 - prediction))

    def evaluate(self,X,Y):
        correct = 0
        for input_ , target in zip(X,Y):  # Using the zip function enables the for loop to take two values each from two different provided list.
            prediction = self.predict(input_)

            if prediction >= 0.5:
                predicted_class = 1
            else:
                predicted_class = 0
            if predicted_class == target :
                correct += 1

        accuracy = correct/len(X)
        return accuracy

# Implementing our perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Important Datasets/diabetes.csv')
print(data.head())

# Checking for null values

print(data.isnull().sum())

# Splitting the data into training and test data

X = data.drop('Outcome',axis=1)
Y = data['Outcome']
print(X.shape)
print(Y.shape)
X = X.values  # Converts dataframe columns into numpy array.
Y = Y.values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42,stratify = Y)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Training our perceptron

model = SigmoidPerceptron(X_train.shape[1])
model.fit(X_train,Y_train,0.1,1000)

# Evaluation of the model
print('--------------------------------------------------------------------------------------')
print('Training data accuracy :: ',model.evaluate(X_train,Y_train))
print('Testing data accuracy :: ',model.evaluate(X_test,Y_test))
