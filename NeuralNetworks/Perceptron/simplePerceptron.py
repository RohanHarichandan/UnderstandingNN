#single layer / simplest neural network 
# - takes multiple inputs 
# - applies weights and bias 
# - uses activation function to decide output

import numpy as np

def step_function(x):
    return 1 if x>=0 else 0

class Perceptron:
    def __init__(self,input_size):
        self.weights=np.random.randn(input_size)
        self.bias=np.random.randn()
    
    def forward(self,inputs):
        weighted_sum=np.dot(inputs,self.weights)+ self.bias
        return step_function(weighted_sum)   

perceptron=Perceptron(input_size=2)
inputs=np.array([1.5,0.5])
output=perceptron.forward(inputs)

print("Perceptron Output:", output)