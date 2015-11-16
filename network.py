class Network(object):
    def __init__(self, layers):
        self.layers = layers

    def predict(self,X):
        Xnext = X
        for layer in self.layers:
            Xnext = layer.getOutput(X)

        return Xnext
