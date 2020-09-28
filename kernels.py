import numpy as np
from abc import ABC, abstractmethod
 
class KernelBase(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def apply(self, x, y):
        pass

class CovKernel(KernelBase):
    def __init__(self, factor):
        super().__init__()
        self.factor=factor

    def apply(self, x, y):
        return np.matrix(self.factor*x@y.T)

class OnesKernel(KernelBase):
    def __init__(self):
        super().__init__()

    def apply(self,x,y=None):
        if y is None:
            return np.ones(x.shape)
        else:
            raise ValueError('OnesKernel: Cannot handle two inputs!')

class ExponentialKernel(KernelBase):
    def __init__(self, lambd, q):
        super().__init__()
        self.lambd=lambd
        self.q=q

    def apply(self,x,y=None):
        if y is None:
            return np.exp((-x**self.q)*self.lambd)
        else:
            raise ValueError('ExponentialKernel: Cannot handle two inputs (NYI)!')        





    


