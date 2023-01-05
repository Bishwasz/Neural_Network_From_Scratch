import gzip
import numpy as np
class Show:
    def showImageNumber(batchSize):
        f = gzip.open('train-images-idx3-ubyte.gz','r')
        image_size = 28
        f.read(16)
        buf = f.read(image_size * image_size * batchSize)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(batchSize, image_size, image_size, 1)

        image = np.asarray(data)
        a=list()
        for i in range(len(image)):
            a.append(image[i].flatten())
        return a

class Label:
    def rOneHot(batchSize):
        f = gzip.open('train-labels-idx1-ubyte.gz','r')
        f.read(8)
        labelValue=list()
        for i in range(batchSize):   
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            labelValue.append(labels[0])
        labelValue=np.array(labelValue)
        oneHot = np.zeros((labelValue.size, labelValue.max() + 1))
        oneHot[np.arange(labelValue.size), labelValue] = 1
        return oneHot