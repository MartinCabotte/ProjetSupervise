import numpy as np

class dataset:
    def __init__(self, dataset):
        self.rought = None
        self.spectral = None
        self.zeroTransform = None

        self.spectralNJW([[1,2,3],[4,5,6],[7,8,9]])

    def spectralNJW(self,W):
        
        #Function inspired by compute.laplacian.NJW (from the sClust R package) based on the NJW
        #algorithm (NIPS 2001)

        degreeMatrix = np.sum(W,axis=1)

        D = np.diag(1/np.sqrt(degreeMatrix))

        Lsym = D @ W @ D

        self.spectral = np.linalg.eig(Lsym)

        