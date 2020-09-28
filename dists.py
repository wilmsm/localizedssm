import numpy as np
from abc import ABC, abstractmethod
 
class DistBase(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def dist(self, x, y):
        pass

class SimpleMatrixDist(DistBase):
    def __init__(self, dist_matrix):
        super().__init__()
        self.dist_matrix=dist_matrix

    def dist(self, x, y):
        return self.dist_matrix[np.ix_(x,y)]

class ShapeNDEuclideanDist(DistBase):
    def __init__(self, points):
        super().__init__()
        self.points=np.array(points)

    def dist(self, x, y):
        return np.sqrt(np.sum((np.repeat(self.points[x,:][:,np.newaxis,:],len(y),axis=1)-np.repeat(self.points[y,:][np.newaxis,:,:],len(x),axis=0))**2,axis=2))
