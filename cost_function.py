import numpy as np
class CostFunction:
    def squaredError(expected, experimental):
        return (experimental-expected)**2
    def crossEntropy(predicted):
        return -np.log(predicted)
    def totalLoss(self,expected,experimental):
        totalLoss=0
        for i in range(len(expected)):
            totalLoss+=self.squaredError(expected[i],experimental[i])
        return totalLoss