import numpy as np
import cPickle
from layers import *
import matplotlib.pyplot as plt
from time import time

DATA = "./data/mnist.pkl"
alpha = 0.01
lam = 0.0001
batch = 100
iters = 20

def exp(data):
    (xtrain,ytrain), valid_set, (xtest,ytest) = data
    xtrain = xtrain.T
    ytrain = ytrain.T
    xtest = xtest.T
    ytest = ytest.T

    d,n = xtrain.shape

    #define network structure
    l1 = Linear(100,d)
    nl1 = Sigmoid()
    l2 = Linear(50,100)
    nl2 = Sigmoid()
    l3 = Linear(10,50)
    nl3 = Sigmoid()
    loss = SoftNLL()


    epoch_acc = np.zeros((iters, xtrain.shape[1]/batch))
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
            o3 = l3.getOutput(s2)
            s3 = nl3.getOutput(o3)
            #y = loss.getOutput(s2, ytrain,[l1.W,l2.W],lam)
            epoch_acc[i,k/batch] = loss.computeAccuracy(s3,ybatch)

            #compute gradients
            d7 = loss.getGradient(s3, ybatch)
            d6 = nl3.getPassback(o3,d7)
            d5 = l3.getPassback(s2,d6)
            d4 = nl2.getPassback(o2, d5)
            d3 =  l2.getPassback(s1, d4)
            d2 = nl1.getPassback(o1,d3)

            l1_gradW, l1_gradb = l1.getGradient(xbatch,lam,d2)
            l2_gradW, l2_gradb = l2.getGradient(s1,lam,d4)
            l3_gradW, l3_gradb = l3.getGradient(s2,lam,d6)


            #update params
            l1.W = l1.W - alpha*(l1_gradW)# + lam*l1.W)
            l1.b = l1.b - alpha*(l1_gradb)
            l2.W = l2.W - alpha*(l2_gradW)# + lam*l2.W)
            l2.b = l2.b - alpha*l2_gradb
            l3.W = l3.W - alpha*(l3_gradW)# + lam*l2.W)
            l3.b = l3.b - alpha*l3_gradb
        print "train epoch: ", i, " accuracy: ", np.mean(epoch_acc[i,:])

        #test
        o1 = l1.getOutput(xtest)
        s1 = nl1.getOutput(o1)
        o2 = l2.getOutput(s1)
        s2 = nl2.getOutput(o2)
        o3 = l3.getOutput(s2)
        s3 = nl3.getOutput(o3)

        a =  loss.computeAccuracy(s3,ytest)
        plot.append((i,a))

        print "test epoch: ",i, " accuracy: ",a

    x, y = zip(*plot)
    plt.plot(range(1,iters+1),y)
    plt.show()
    f = open('plot_'+str(time())+'.txt','w')
    for i,v in plot:
        f.write(str(i) + "," + str(v) + "\n")

    f.close()
def load_data():
    f = open(DATA)
    data = cPickle.load(f)
    f.close()
    return data

if __name__ == "__main__":
    data = load_data()
    exp(data)
