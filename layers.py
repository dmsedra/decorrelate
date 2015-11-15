import numpy as np
import numpy.random as rand


class Linear(object):
    def __init__(self,n_out, n_in):
        self.W = rand.normal(0.0,2.0/(n_in+n_out),(n_out,n_in)) #use Xavier initialization
        self.b = 0.01*np.ones((n_out,1))

    def getOutput(self,X):
        return np.dot(self.W,X) + self.b

    # passback - (n_out,n)
    def getGradient(self,X,passback):
        gradW = np.dot(passback, np.transpose(X))
        gradb = np.sum(passback,1)
        gradb = gradb[:,np.newaxis] #keep from broadcasting
        return (gradW, gradb)

    def getPassback(self,X,passback):
        return np.dot(self.W.T, passback)

class Sigmoid(object):
    def __init__(self):
        pass

    def getOutput(self,X):
        return 1/(1+np.exp(-X))

    def getPassback(self,X,passback):
        local = np.multiply(self.getOutput(X), 1-self.getOutput(X))
        return np.multiply(local,passback)


class SquaredLoss(object):
    def __init__(self):
        pass

    def getOutput(self,X,Y):
        return 0.5*np.linalg.norm(X-Y)**2

    def getGradient(self,X,Y):
        return X-Y

class  Covariance(object):
    def __init__(self):
        pass

    def getOutput(self,X):
        Xcov = np.dot(X, X.T)

        d,V = np.linalg.eigh(Xcov)

        D = np.diag(1./np.sqrt(d+1e-5))

        Rot = np.dot(np.dot(V, D), V.T)

        X_white = np.dot(Rot, X)
        return X_white

    def getPassback(self,X,passback):
        Xcov = np.dot(X, X.T)

        d,V = np.linalg.eigh(Xcov)

        D = np.diag(1./np.sqrt(d+1e-5))

        Rot = np.dot(np.dot(V, D), V.T)

        return np.dot(Rot.T, passback)


class SoftNLL(object):
    def __init__(self):
        pass


    #compute SoftMax + NLL
    def getOutput(self,X,Y):
        def colSoft(col):
            e = np.exp(col)
            return e/np.sum(e)

        sm = np.apply_along_axis(colSoft, 0, X)
        res = []
        for i,v in enumerate(Y):
            res.append(sm[v,i])

        return np.mean(-1*np.log(res))


    def getGradient(self,X,Y):
        def colSoft(col):
            e = np.exp(col)
            return e/np.sum(e)

        sm = np.apply_along_axis(colSoft, 0, X)
        for i,v in enumerate(Y):
            sm[v,i] = sm[v,i]-1

        return sm





