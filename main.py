from network import NeuralNetwork
tes=NeuralNetwork(inputSize=784,numOfNode=100,outPutLayerNodes=10,hiddenLayer=1)
tes.learn(trainingSize=500,batchSize=90,learnRate=0.5)
