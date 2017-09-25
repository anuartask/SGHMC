import numpy as np
from optimizers import SGD

class SGHMC(object):
    
    def __init__(self,
                 grad_objective,
                 eta=0.01, batch_size=100,
                 epochs=10, seed=None, 
                 alpha=0):
        
        self.optimizer = SGD(grad_objective, 
                             eta=eta, 
                             epochs=epochs,
                             batch_size=batch_size,
                             seed=seed,
                             alpha=1. - alpha)
        self.batch_size = batch_size
        self.noise_var = np.sqrt(2 * eta * alpha)
        self.epochs = epochs
        
    def samples_return(self, X, w_init=None, y=None, display=False):
        self.num_objects, num_features = X.shape
        self.batch_counts = self.num_objects // self.batch_size
        indices = np.arange(num_objects)
        iterations_cnt = self.batch_counts * self.epochs
        self.samples = np.zeros((iterations_cnt, num_features))
        noise = self.noise_var * np.random.randn(iterations_cnt, num_features)
        for epoch in range(self.epochs):
            mini_batches = np.random.choice(indices, size=self.num_objects, replace=False)
            for batch in range(self.batch_counts):
                batch_ind = mini_batches[batch * self.batch_size: (batch + 1) * self.batch_size]
                X_batch = X[batch_ind, :]
                if y is not None:
                    y_batch = y[batch_ind]
                else:
                    y_batch = None
                i = epoch * self.batch_counts + batch
                self.optimizer.w = self.optimizer.step(X_batch, w_init=w_init, y=y_batch, 
                                                       batch_counts=self.batch_counts)
                self.optimizer.w += noise[i, :]
                self.samples[i, :] = self.optimizer.w
            if display and (epoch % 100 == 0):
                print(epoch)                          
        return self.samples