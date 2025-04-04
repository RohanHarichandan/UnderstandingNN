import numpy as np 
import matplotlib.pyplot as plt

#step function
def step(x):
    return np.where(x>1,1,0)

#sigmoid 
def sigmoid(x):
    return 1/(1+np.exp(-x))

#tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

#ReLU
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)

def leaky_relu(x,alpha=0.01):
    return np.where(x>0,x,alpha*x)

def leaky_relu_derivative(x,alpha=0.01):
    return np.where(x>0,1,alpha)

#softmax
def softmax(x):
    exp_x=np.exp(x-np.max(x))
    return exp_x/np.sum(exp_x,axis=0)


#visualizing the activation functions 
x=np.linspace(-5,5,100)
plt.figure(figsize=(10,6))
plt.plot(x,step(x),label="step",linestyle='dotted')
plt.plot(x,sigmoid(x),label="Sigmoid")
plt.plot(x,tanh(x),label="Tanh")
plt.plot(x,relu(x),label="ReLU")
plt.plot(x,leaky_relu(x),label="Leaky ReLU")
plt.legend()
plt.title("Activation Functins")
plt.grid()
plt.show()
