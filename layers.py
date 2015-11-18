import numpy as np
import numpy.random as rand


class Linear(object):
    def __init__(self,n_out, n_in):
        self.W = rand.normal(0.0,0.2,(n_out,n_in)) #use Xavier initialization
        self.b = np.zeros((n_out,1))
        self.Wvel = np.zeros((n_out,n_in))

    def getOutput(self,X):
        out = np.dot(self.W,X) + self.b
        return out

    # passback - (n_out,n)
    def getGradient(self,X,lam,passback):
        m = X.shape[1]
        gradW = np.dot(passback, np.transpose(X))

        gradb = np.sum(passback,1)
        gradb = gradb[:,np.newaxis] #keep from broadcasting
        gradW, gradb = (gradW/m + lam*self.W, gradb/m)
        self.gradW = gradW
        self.gradb = gradb
        return (gradW,gradb)

    def updateMom(self,X,alpha,gamma):
        self.Wvel = alpha*self.gradW+gamma*self.Wvel

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
        comp = np.zeros(X.shape)
        for i in range(X.shape[1]):
            comp[Y[i],i] = 1.0
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

class ReLU(object):
    def __init__(self):
        pass

    def getOutput(self,X):
        return np.maximum(0,X)

    def getPassback(self,X,passback):
        dx = np.zeros(X.shape)
        dx[X >= 0] = 1
        return np.multiply(dx,passback)


class SoftNLL(object):
    def __init__(self):
        pass

    def getOutput(self,X):
        sm = np.exp(X - np.amax(X,axis=0))
        sm = sm/np.sum(sm,0)
        return sm

    def getGradient(self,X,Y):
        Y_pred = self.getOutput(X)
        for i in range(len(Y)):
            Y_pred[Y[i],i] -= 1
        return Y_pred

    #compute SoftMax + NLL
    def getLoss(self,X,Y,Weights,lam):
        sm = np.exp(X - np.amax(X,axis=0))
        sm = sm/np.sum(sm,0)
        res = []
        for i,v in enumerate(Y):
            res.append(sm[v,i])

        L2 = np.sum(map(lambda z: np.linalg.norm(z,2)**2, Weights))
        return np.sum(-1*np.log(res)) + lam*L2/2.0

    def computeAccuracy(self,X,Y):
        sm = np.apply_along_axis(np.argmax, 0, X)
        accuracy = 0.0
        m = X.shape[1]

        for i,v  in enumerate(sm):
            if sm[i] == Y[i]:
                accuracy = accuracy + 1

        return accuracy/m

#    def getGradient(self,X,Y):
#        sm = np.exp(X - np.amax(X,axis=0))
#        sm = sm/np.sum(sm,0)
#
#        for i,v in enumerate(Y):
#            sm[v,i] = sm[v,i]-1
#
#        return sm

    def gradCheck(self,X,Y,Weights,lam):
        verify = self.getGradient(X,Y)
        eps = 1e-5
        (n,m) = X.shape
        check = np.zeros(X.shape)
        for i in range(n):
            for j in range(m):
                X1 = np.copy(X)
                X1[i,j] = X[i,j] + eps
                X2 = np.copy(X)
                X2[i,j] = X[i,j] - eps
                check[i,j] = (self.getOutput(X1,Y,Weights,lam)-self.getOutput(X2,Y,Weights,lam))/(2*eps)

        return np.linalg.norm(verify-check)


