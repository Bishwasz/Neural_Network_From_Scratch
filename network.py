import numpy as np

from activations import Activation
from cost_function import CostFunction
from derivatives import Derivative
from Get_Data import Label, Show
from layer import Layer

import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

class NeuralNetwork:
    def __init__(self,inputSize,numOfNode,outPutLayerNodes,hiddenLayer):
        self.outPutLayerSize=outPutLayerNodes
        self.numOfNode=numOfNode
        self.inputSize=inputSize
        self.hiddenLayer=hiddenLayer
        self.activationOutput=np.array([[0]*self.inputSize if i==0 else [0]*self.numOfNode if (i<self.hiddenLayer+1 and i!=0) else [0]*10 for i in range(self.hiddenLayer+2)],dtype=object)
        self.layers=[Layer(inputSize,self.numOfNode) if i==0 else Layer(self.numOfNode,self.outPutLayerSize) if i==(self.hiddenLayer) else Layer(self.numOfNode,self.numOfNode)for i in range(self.hiddenLayer+1)]

    def networkOutput(self,input):
        self.activationOutput[0]=input
        for i in range(self.hiddenLayer+1):
            input=self.layers[i].output(input)
            self.activationOutput[i+1]=input
        bra=Activation.softMaxActivationWMaxP(input)
        return (bra)

    def getDelta(self,softOut,oHot):
        delta=[list()]
        for i in range(self.outPutLayerSize):
            de=(softOut[i]-oHot[i])*Derivative.dRelU(self.activationOutput[self.hiddenLayer+1][i])
            delta[0].append(de)
        for i in range(self.hiddenLayer-1,-1,-1):
            delta.append([])
            for z in range(self.layers[i].outgoing):
                de=0
                for a in range(self.layers[i+1].outgoing):
                    wait=self.layers[i+1].matrixConnection[z][a]
                    activationDer=Derivative.dRelU(self.activationOutput[i+1][z])
                    d=delta[self.hiddenLayer-i-1][a]
                    de+=d*wait*activationDer
                delta[len(delta)-1].append(de)
        delta.reverse()
        return delta

    def backPropogation(self,softOut,oHot,delta):
        for i in range(self.hiddenLayer+1):
            for k in range(self.layers[i].outgoing):
                for j in range(self.layers[i].incoming):
                    d=delta[i][k]*self.activationOutput[i][j]
                    # print(d)
                    self.layers[i].gradient[j][k]+=d
                self.layers[i].gradientBias[k]+=delta[i][k]

    def clearGradients(self):
        for i in range(self.hiddenLayer+1):
            self.layers[i].clear_Gradient()

    def updategGradient(self,lRate,batchSize):
        for i in range(self.hiddenLayer+1):
            for k in range(self.layers[i].outgoing):
                for j in range(self.layers[i].incoming):
                    self.layers[i].matrixConnection[j][k]+=-lRate*(self.layers[i].gradient[j][k]/batchSize)
                self.layers[i].bias[k]+=-lRate*(self.layers[i].gradientBias[k]/batchSize)

    def learn(self, trainingSize,batchSize,learnRate):
        img=Show.showImageNumber(trainingSize)
        ans=Label.rOneHot(trainingSize)
        progress=list()
        xAxis=list()
        totalLoss=list()
        a=plt.subplot()

        loc = plticker.MultipleLocator(base=2.0)
        for i in range(trainingSize//batchSize):
            succes=0
            loss=0
            for x in range(batchSize):
                exp=self.networkOutput(img[i*batchSize+x])
                delta=self.getDelta(exp,ans[i*batchSize+x])
                loss+=CostFunction.crossEntropy(exp[ans[i*batchSize+x].argmax()])
                if(exp.argmax()==ans[i*batchSize+x].argmax()):
                    succes+=1
                self.backPropogation(exp,ans[i*batchSize+x],delta)
            print(succes)
            succesRate=(succes/batchSize)*100
            totalLoss.append(succesRate)
            progress.append(succesRate)
            xAxis.append(i)

            self.updategGradient(learnRate,batchSize)
            self.clearGradients()
        a.plot(progress)
        # a.xaxis.set_major_locator(ticker.MultipleLocator(2))
        # a.plot(xAxis,totalLoss,color='r')
        # a.xticks(np.arange(0,max(xAxis)+1,2))
        a.axis(ymin=0,ymax=100)
        # a.set_xticks()
        a.set_ylabel("Succes Rate (%)")
        a.set_xlabel("Number of Batches Trained")
        a.set_title("Training Accuracy")
        # a.set_xticks(np.arange(0,max(xAxis)+1,2))
     
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
        plt.show()