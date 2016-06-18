def read_CIFAR(CIFARdir):
    import os
    import numpy as np

    Xtr = np.zeros((1, 3072))
    Ytr = np.array([])
    Xte = np.array((1, 3072))
    Yte = np.array([])

    for file in os.listdir(CIFARdir):
        filename = CIFARdir + '/' + file
        
        if(file.find("data_batch") >= 0):
            X = unpickle(filename)
            print filename 
            Xtr = np.concatenate((Xtr, X['data']))
            Ytr = np.concatenate((Ytr, X['labels']))
        elif(file.find("test_batch") >= 0):
            X = unpickle(filename)
            print filename
            Xte = np.concatenate((Xtr, X['data']))
            Yte = np.concatenate((Ytr, X['labels']))

    return Xtr, Ytr, Xte, Yte

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#(Xtr, Ytr, Xte, Yte) = read_CIFAR("/home/ltq/data.mining/cifar-10-batches-py")

#print Xtr.shape

