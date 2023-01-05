import numpy as np
from activations import Activation
class Layer:
    def __init__(self, incoming,outgoing):
        self.incoming=incoming
        self.outgoing=outgoing
        # Gradients For Weights And Biases
        self.gradient=np.array(0*np.random.randn(self.incoming,self.outgoing),dtype=np.float64)
        self.gradientBias=np.array([0 for i in range(self.outgoing)],dtype=np.float64)

        # Weights And Biases
        self.matrixConnection=np.array(0.1*np.random.randn(self.incoming,self.outgoing),dtype=np.float64)
        self.bias=np.array(np.random.randn(self.outgoing),dtype=np.float64)

    def output(self,input):
        # The Output of a Network
        outPutMatrix=input@self.matrixConnection
        for i in range(self.outgoing):
            outPutMatrix[i]+=self.bias[i]
            outPutMatrix[i]=Activation.sigmoid(outPutMatrix[i])

        return outPutMatrix
    def clear_Gradient(self):
        self.gradient=np.array(0*np.random.randn(self.incoming,self.outgoing),dtype=np.float64)
        self.gradientBias=np.array([0 for i in range(self.outgoing)],dtype=np.float64)