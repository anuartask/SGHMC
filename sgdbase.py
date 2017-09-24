from abc import ABCMeta, abstractmethod

class SGDBase(metaclass=ABCMeta):
    """
        Abstract class for stochastic optimization methods.
        
        Parameters:
        grad_objective: func, gradient of minimized objective function.
        eta: float, learning rate parameter.
        batch_size: int, size of mini-batches.
        epochs: int, epochs size.
    """
    def __init__(self,
                 grad_objective,
                 eta=0.1, batch_size=100,
                epochs=10, seed=None):
        self.grad_objective = grad_objective
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        
    @abstractmethod
    def _next_direction(self, X, y=None):
        """
            This method determine the next direction of method.
        """
        pass
    
    
    @abstractmethod
    def step(self, X, y=None):
        """
            This method do one step to minimize objective function.
        """
        pass
    
    
    @abstractmethod
    def train(self, X, y=None):
        """
            This method converges to the local minima of objective.
        """
        pass