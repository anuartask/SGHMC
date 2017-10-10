import numpy as np

class SGHMC(object):
    
    def __init__(self, grad_objective,
                eta=1e-4, batch_size=100,
                epochs=10, seed=None, 
                 alpha=0.01):
        self.grad_objective = grad_objective
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.alpha = alpha
        self.noise_var = np.sqrt(2 * eta * alpha)
    
    def samples_return(self, X, w_init=None, y=None, display=False):
        
        np.random.seed(self.seed)
        self.num_objects, num_features = X.shape
        self.batch_counts = self.num_objects // self.batch_size
        self._batch_counts = self.num_objects / self.batch_size
        indices = np.arange(num_objects)
        iterations_cnt = self.batch_counts * self.epochs
        self.samples = np.zeros((iterations_cnt, num_features))
        noise = self.noise_var * np.random.randn(iterations_cnt, num_features)
        self.velocity = np.zeros(num_features)
        
        if w_init is not None:
            self._w = w_init
        else:
            self._w = np.zeros(num_features)
        
        for epoch in range(self.epochs):
            mini_batches = np.random.permutation(indices)
            for batch in range(self.batch_counts):
                i = epoch * self.batch_counts + batch
                batch_ind = mini_batches[batch * self.batch_size: (batch + 1) * self.batch_size]
                X_batch = X[batch_ind, :]
                if y is not None:
                    y_batch = y[batch_ind]
                else:
                    y_batch = None
                if y is not None:
                    new_w = self._w - self._batch_counts * self.eta * self.grad_objective(self._w, X_batch, y_batch) +\
                            (1. - self.alpha) * self.velocity +\
                            noise[i, :]
                    
                    self.velocity = new_w - self._w
                    self._w = new_w.copy()
                else:
                    new_w = self._w - self._batch_counts * self.eta * self.grad_objective(self._w, X_batch) +\
                            (1. - self.alpha) * self.velocity +\
                            noise[i, :]
                    self.velocity = new_w - self._w
                    self._w = new_w.copy()
                
                self.samples[i, :] = self._w
            if display and (epoch % 100 == 0):
                print(epoch)                          
        return self.samples

class SGNHT(object):    
    def __init__(self,
                 grad_objective,
                 eta=0.01, batch_size=100,
                 epochs=10, seed=None, 
                 a=0):
        self.grad_objective = grad_objective
        self.eta = eta
        self.epochs = epochs
        self.alpha = a
        self.noise_var = np.sqrt(2 * eta * a)
        self.batch_size = batch_size
        self.seed = seed
        self.epochs = epochs
    
    def samples_return(self, X, w_init=None, y=None, display=False):
        np.random.seed(self.seed)
        self.num_objects, num_features = X.shape
        self.batch_counts = self.num_objects // self.batch_size
        self._batch_counts = self.num_objects / self.batch_size
        
        indices = np.arange(num_objects)
        iterations_cnt = self.batch_counts * self.epochs
        self.samples = np.zeros((iterations_cnt, num_features))
        noise = self.noise_var * np.random.randn(iterations_cnt, num_features)
        self.velocity = np.zeros(num_features)
        
        self.alphas = []
        
        if w_init is not None:
            self._w = w_init
        else:
            self._w = np.zeros(num_features)
        
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
                
                if y is not None:
                    new_w = self._w - self.eta * self._batch_counts * self.grad_objective(self._w, X_batch, y_batch) +\
                            (1. - self.alpha) * self.velocity +\
                            noise[i, :]
                    self.velocity = new_w - self._w
                    self._w = new_w.copy()
                    self.alphas.append(self.alpha)
                    self.alpha += 1. / num_features * (self.velocity ** 2).sum() - self.eta
                else:
                    new_w = self._w - self.eta * self._batch_counts * self.grad_objective(self._w, X_batch) +\
                            (1. - self.alpha) * self.velocity +\
                            noise[i, :]
                    self.velocity = new_w - self._w
                    self._w = new_w.copy()
                    self.alphas.append(self.alpha)
                    self.alpha += 1. / num_features * (self.velocity ** 2).sum() - self.eta
                
                self.samples[i, :] = self._w
            if display and (epoch % 100 == 0):
                print(epoch)                          
        return self.samples