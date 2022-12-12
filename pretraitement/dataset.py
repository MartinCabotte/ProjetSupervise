import numpy as np
import pandas as pd

class dataset:
    def __init__(self, dataset, vois):

        #Dataset features are represented by columns and rows represente samples

        self.rought = pd.DataFrame(dataset)
        self.spectral = None
        self.similarityZP = None
        self.zeroTransform = None

        self.computeSpectralEmbeddedSpace(self.rought,vois)

    def spectralNJW(self):
        
        #Function inspired by compute.laplacian.NJW (from the sClust R package) based on the NJW
        #algorithm (NIPS 2001)

        W = self.similarityZP

        degreeMatrix = np.sum(W,axis=1)

        D = np.diag(1/np.sqrt(degreeMatrix))

        Lsym = D @ W @ D

        self.spectral = np.linalg.eig(Lsym)

        print("-- Spectral NJW is computed --")

    def ZPSimilarityMatrix(self,data,vois):

        #Function inspired by compute.similarity.ZP (from the sClust R package) 

        #Check if the number of points is enought in the dataset
        vois = min(vois, data.shape[0]-1)
        print("Min points is : ",vois)

        #Compute distance matrix using euclidian metric
        distMatrix = []

        #Loop over all points
        for i in range(data.shape[0]):
            #For each point, compute distance between this points and all others
            distMatrix.append([(np.sqrt(np.sum((data.iloc[i] - data) ** 2, axis=1)))])
        distMatrix = pd.DataFrame(distMatrix)

        ###search.neightboor() from sClust R package
        #Returns the value of the i-th neighboor of each point

        sigmaV = []
        for i in range(distMatrix.shape[0]):
            sortedRow = sorted(distMatrix.iloc[i].tolist()[0])
            sigmaV.append(sortedRow[vois])

        sigma = []
        for i in range(distMatrix.shape[0]):
            sigma.append((np.array(sigmaV)*np.array([sigmaV[i]]*distMatrix.shape[0])).tolist())

        ##Compute matrix multiplication of distMatrix (E^2) in the final formula
        E2 = np.zeros((distMatrix.shape[0],distMatrix.shape[0]),dtype = float)
        for i in range(distMatrix.shape[0]):
            for j in range(distMatrix.shape[0]):
                E2[i,j] += distMatrix.iloc[i].tolist()[0][j] ** 2

        #Finally, we compute similarity Matrix according to Zelnik and Perona
        self.similarityZP = np.exp(-(E2)/sigma)

        print("-- Similarity matrix (ZP) is completed --")
        

    def checkGramMatrix(self):
        
        #Function inspired by checking.gram.similarityMatrix (from the sClust R package) 

        W = self.similarityZP

        v,w = np.linalg.eig(W)

        if(len(v<0)>0):
            print("-- Non-Gram Matrix --")
            W = W @ W.T

        np.fill_diagonal(W,0)

        self.similarityZP = W

    def computeSpectralEmbeddedSpace(self,data,vois=3):
        
        W = None
        self.ZPSimilarityMatrix(data,vois)
        self.checkGramMatrix()
        self.spectralNJW()

        v,w = self.spectral

        indexes = np.where(v > 0.1)[0]

        self.spectral = pd.DataFrame(w[:,indexes])

        print(self.spectral)