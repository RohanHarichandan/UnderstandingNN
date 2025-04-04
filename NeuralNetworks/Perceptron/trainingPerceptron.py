import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return 1 if x>=0 else 0

class Perceptron:
    def __init__(self,input_size,learning_rate=0.1):
        self.weights=np.random.randn(input_size)
        self.bias=np.random.randn()
        self.learning_rate=learning_rate
    
    def forward(self,inputs):
        weighted_sum=np.dot(inputs,self.weights)+self.bias
        return step_function(weighted_sum)
    
    def train(self,X,y,epochs=10):
        for epoch in range(epochs):
            total_error=0
            for i in range(len(X)):
                prediction=self.forward(X[i])
                error=y[i]-prediction
                total_error+=abs(error)
                self.weights+=self.learning_rate*error*X[i]
                self.bias+=self.learning_rate*error
            print(f"Epoch {epoch+1}/{epochs}, Total Error: {total_error}")    
#generate random data 
np.random.seed(42)
X=np.random.randn(100,2)  
y=np.array([1 if x[0] + x[1]>0 else 0 for x in X])  

#train perceptron
perceptron=Perceptron(input_size=2,learning_rate=0.1)
perceptron.train(X,y,epochs=10)

#predict the final output

prediction=np.array([perceptron.forward(x) for x in X])


#Plot decision boundary

plt.figure(figsize=(8,2))
plt.scatter(X[:,0], X[:,1], c=prediction, cmap="coolwarm", edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary after Training")
plt.colorbar(label="Predicted Class")
plt.show()                    