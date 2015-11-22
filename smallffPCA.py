import cPickle
import matplotlib.pyplot as plt
from time import time

import numpy as np
from scipy import stats

from layers import *

DATA = "./data/mnist.pkl"
alpha = 0.05
lam = 0.001
gamma = 0.95
batch = 100
train = 2000
test = 300
iters = 35

def exp(data):
    (xtrain,ytrain), valid_set, (xtest,ytest) = data
    xtrain = zscore(xtrain[:train,:].T)
    ytrain = ytrain[:train]
    xtest = zscore(xtest[:test,:].T)
    ytest = ytest[:test]

    d,n = xtrain.shape

    #define network structure
    l1 = Linear(800,d)
    nl1 = Sigmoid()
    l2 = Linear(10,800)
    nl2 = Sigmoid()
    loss = SoftNLL()


    epoch_acc = np.zeros((iters, xtrain.shape[1]/batch))
    epoch_loss = np.zeros((iters, xtrain.shape[1]/batch))

    plot = []
    #train
    for i in range(iters):
        for k in range(0,xtrain.shape[1],batch):
            xbatch = xtrain[:,k:k+batch]
            ybatch = ytrain[k:k+batch]
            #forward pass
            o1 = l1.getOutput(xbatch)
            s1 = nl1.getOutput(o1)
            o2 = l2.getOutput(s1)
            s2 = nl2.getOutput(o2)
            #y = loss.getOutput(s2, ytrain,[l1.W,l2.W],lam)
            epoch_acc[i,k/batch] = loss.computeAccuracy(s2,ybatch)
            epoch_loss[i,k/batch] = loss.getLoss(s2,ybatch,[l1.W,l2.W],lam)

            #compute gradients
            d5 = loss.getGradient(s2, ybatch)
            d4 = nl2.getPassback(o2, d5)
            d3 =  l2.getPassback(s1, d4)
            d2 = nl1.getPassback(o1,d3)

            l1_gradW, l1_gradb = l1.getGradient(xbatch,lam,d2)
            l2_gradW, l2_gradb = l2.getGradient(s1,lam,d4)


            l1.updateMom(xbatch,alpha,gamma)
            l2.updateMom(s1,alpha,gamma)


            #update params
            l1.W = l1.W - l1.Wvel# + lam*l1.W)
            l1.b = l1.b - alpha*(l1_gradb)
            l2.W = l2.W - l2.Wvel
            l2.b = l2.b - alpha*l2_gradb

        #test
        o1 = l1.getOutput(xtest)
        s1 = nl1.getOutput(o1)
        o2 = l2.getOutput(s1)
        s2 = nl2.getOutput(o2)
        a =  loss.computeAccuracy(s2,ytest)
        plot.append((i,a))

        print "epoch: ",i, " test accuracy: ",a," train accuracy: ",np.mean(epoch_acc[i,:])," train loss: ",np.mean(epoch_loss[i,:])

    x, y = zip(*plot)
    plt.plot(range(1,iters+1),y)
    plt.plot(range(1,iters+1),np.mean(epoch_acc,1))
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()
    f = open('plot_'+str(time())+'.txt','w')
    for i,v in plot:
        f.write(str(i) + "," + str(v) + "\n")

    f.close()

#row wise examples
def zscore(data):
    a,b = data.shape
    return (data-np.mean(data,1).reshape((a,1)))/(np.std(data,1).reshape((a,1))+1e-8)

def load_data():
    f = open(DATA)
    data = cPickle.load(f)
    f.close()
    return data

if __name__ == "__main__":
    data = load_data()
    exp(data)
