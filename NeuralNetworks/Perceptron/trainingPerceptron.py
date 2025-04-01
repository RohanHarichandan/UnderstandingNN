import numpy as np
import matplotlib as plt

def step_function(x):
    return 1 if x>=0 else 0

class Perceptron:
    def __init__(self,input_size,learning_rate+0.1):
        self.weights=np.random.randn(input_size)
        self.bias=np.ramdom.randn()
        self.learning_rate=learning_rate
    
    def forward(self,inputs):
        weighted_sum=np.dot(inputs,self.weights)+self.bias
        return step_function(weighted_sum)
    
    def train(self,X,y,epochs=10):
        for epoch in range(epochs):
            for i in range(len(X)):
                prediction=self.forward(X[i])
                error=y[i]-prediction
                self.weights+=self.learning_rate*error*X[i]
                self.bias+=self.learning_rate*error
            