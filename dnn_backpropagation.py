import numpy as np

def sigmoid(x): return 1./(1+np.exp(-x))
def dsigmoid(x): return sigmoid(x)*(1-sigmoid(x))

class dnn:
    def __init__(self, nodes = [5,5,1]):
        self.weights = [np.random.random_sample((m,n)) for m,n in zip(nodes[:-1],nodes[1:])]
        self.deltas = [np.zeros((m,n)) for m,n in zip(nodes[:-1],nodes[1:])]
        self.errors = [np.zeros(n) for n in nodes]
        self.nodes = [np.zeros(n) for n in nodes]
        self.fnodes = [np.zeros(n) for n in nodes]
        self.f = sigmoid
        self.df = dsigmoid
        self.Lambda = 1
        self.m = 1
        self.Alpha = 0.1

    def forward(self,X):
        """
        X: np.array([...])
        """
        self.nodes[0] = X
        self.fnodes[0] = self.f(X)
        for j,w in enumerate(self.weights):
            temp = self.fnodes[j]@w
            self.nodes[j+1] = temp
            self.fnodes[j+1] = self.f(temp)

    def backward(self,y):
        """
        y: np.array([...])
        """
        #Error recursion
        self.errors[-1] = -(self.fnodes[-1] - y)
        for j,w in reversed(list(enumerate(self.weights))):
            self.errors[j] = (self.errors[j+1]@w.T)*self.df(self.nodes[j])

        #Computing deltas and updating weights
        for l,delta in enumerate(self.deltas):
            delta += self.fnodes[l].reshape((self.fnodes[l].shape[0],1))*self.errors[l+1]
            self.weights[l] += (self.Alpha/self.m)*delta

    def decide(self,X):
        self.forward(X)
        p = self.fnodes[-1]
        return np.round(p)

"""
TODO:
41  Add regularization
init Add bias
"""
