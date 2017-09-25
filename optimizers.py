import numpy as np
from sgdbase import SGDBase

class SGD(SGDBase):
    
    def __init__(self,
                 grad_objective,
                 eta=0.01, batch_size=100,
                 epochs=10, seed=None, 
                 alpha=0, save_trajectory=False):
        
        SGDBase.__init__(self, 
                         grad_objective,
                         eta=eta,
                         batch_size=batch_size,
                         epochs=epochs,
                         seed=seed)
        self.alpha = alpha
        self.save_trajectory = save_trajectory
        if self.save_trajectory:
            self.w_values = []
        self._trained = False
    
    def _next_direction(self, X, y=None):
        
        res = self.alpha * self.velocity
        
        if y is None:
            return res - self.eta * self.batch_counts * self.grad_objective(self.w, X)
        else:
            return res - self.eta * self.batch_counts * self.grad_objective(self.w, X, y)
        
    
    def step(self, X, w_init=None, y=None, batch_counts=None):
        self.num_objects, num_features = X.shape
        if not self._trained:
            self.batch_counts = batch_counts
            if w_init is not None:
                self.w = w_init
            else:
                self.w = np.zeros(num_features)
            self.velocity = np.zeros(num_features)
            self._trained = True
        self.velocity = self._next_direction(X, y=y)
        self.w = self.w + self._next_direction(X, y=y)
        if self.save_trajectory:
            self.w_values.append(self.w)
        return self.w
    
    
    def train(self, X, y=None, w_init=None, display=False):
        
        self.num_objects, num_features = X.shape
        self.batch_counts = self.num_objects // self.batch_size
        indices = np.arange(num_objects)
        self._trained = True
        if w_init is None:
            self.w = np.zeros(num_features)
        else:
            self.w = w_init
        self.velocity = np.zeros(num_features)
        for epoch in range(self.epochs):
            mini_batches = np.random.choice(indices, size=self.num_objects, replace=False)
            for batch in range(self.batch_counts):
                batch_ind = mini_batches[batch * self.batch_size: (batch + 1) * self.batch_size]
                X_batch = X[batch_ind, :]
                if y is not None:
                    y_batch = y[batch_ind]
                else:
                    y_batch = None
                self.w = self.step(X_batch, y=y_batch)
                
            if display and (epoch % 100 == 0):
                print(epoch)
                    
        return