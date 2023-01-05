import numpy as np
class Activation:
    def relU(x):
        return max(0,x)
    def sigmoid(gamma):
        if gamma < 0:
            return 1 - 1/(1 + np.exp(gamma))
        else:
            return 1/(1 + np.exp(-gamma))
    def hyperbolicTan(x):
        return np.tanh(x)
    def softMaxActivationWMaxP(arrX):
        intermediate=np.exp(arrX-np.max(arrX))
        distribution=intermediate/np.sum(intermediate)
        return distribution

    def softMax(numerator, demoninator):
        return np.exp(numerator)/np.sum(np.exp(demoninator))