import activations as Activation
class Derivative:
    def dSigmoid(x):
        a=Activation.sigmoid(x)
        return a*(1-a)
    def dRelU(x):
        if x<0:
            return 0
        else:
            return 1
    def dMeanS(predicted, actual):
        return 2*(actual-predicted)