import numpy as np
import cPickle
from layers import *

DATA = "./data/mnist.pkl"
alpha = 0.01
lam = 0.01
m = 1000

def train(data):
    (xtrain,ytrain), valid_set, test_set = data
    xtrain = np.transpose(np.asarray(xtrain)[:m])
    ytrain = np.asarray(ytrain)[:m]

    d,n = xtrain.shape

    #define network structure
    l1 = Linear(100,d)
    nl1 = Sigmoid()
    cs1 = Covariance()
    l2 = Linear(10,100)
    nl2 = Sigmoid()
    loss = SoftNLL()

    for i in range(1000):
        #forward pass
        o1 = l1.getOutput(xtrain)
        s1 = nl1.getOutput(o1)
        c1 = cs1.getOutput(s1)
        o2 = l2.getOutput(c1)
        s2 = nl2.getOutput(o2)
        y = loss.getOutput(s2, ytrain,lam)

        print "step: ",i," loss: ",y, " accuracy: ", loss.computeAccuracy(s2, ytrain)
        #compute gradients
        d5 = loss.getGradient(s2, ytrain,[l1.W,l2.W])
        d4 = nl2.getPassback(o2, d5)
        d3 = l2.getPassback(c1, d4)
        d2 =  cs1.getPassback(s1, d3)
        d1 = nl1.getPassback(o1, d2)

        l1_gradW, l1_gradb = l1.getGradient(xtrain,lam,d1)
        l2_gradW, l2_gradb = l2.getGradient(s1,lam,d4)


        print l1_gradW[:,:10]
        #update params
        l1.W = l1.W - alpha*(l1_gradW)# + lam*l1.W)
        l1.b = l1.b - alpha*(l1_gradb)
        l2.W = l2.W - alpha*(l2_gradW)# + lam*l2.W)
        l2.b = l2.b - alpha*l2_gradb

def load_data():
    f = open(DATA)
    data = cPickle.load(f)
    f.close()
    return data

if __name__ == "__main__":
    data = load_data()
    train(data)
