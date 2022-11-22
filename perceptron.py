import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier

from sklearn.datasets import make_classification


class PerceptronClassifier:
    
    def __init__(self,mode=0,CrossValidation=0):
        
        
        self.lamb = 0.0001
        self.learningRate = 0.01
        self.w = np.array([0,5]) #parametre aléatoire
        self.w0 = -10 #parametre aleatoire
        self.mode = mode #determination du model : 0 "OneVsRest" ; 1 : "OneVsOne"
        self.CrossValidation = CrossValidation #determination de l'utilisation de scikit pour la cross validation : 0 sans, 1 avec
        self.model = ""
        
    def entrainement(self,data_train,target_train):
        
        per = Perceptron(alpha=self.lamb,eta0=self.learningRate,penalty="l2")
        
        if self.mode == 0:
        
            model = OneVsRestClassifier(per)
            
        else : 
            
            model = OneVsOneClassifier(per)
            
        self.model = model.fit(data_train,target_train)

    def validation_croisee(self,data,target):
        
        K = 10
        bestError = -1
        bestLambda = 0
        bestLearningRate = 0
        
        #on mélange tout d'abord nos données
        index = np.arange(0,len(data),1)
        np.random.shuffle(index)

        data = data[index]
        target = target[index]


        #On les divise en k paquets
        paquetsX = np.array_split(data,K)
        paquetst = np.array_split(target,K)
        
        
        
        #on réalise les simulations
        for lambda_test in np.arange(0.1,1,0.1):
            self.lamb = lambda_test
            print(self.lamb)
            
            meanError = 0
            for learningRate_test in np.arange(0.01,1,0.01):
                self.learningRate = learningRate_test
                
                for i in range(0,K):
                    
                    testX = paquetsX[i]
                    testT = paquetst[i]
                    
                    validationX = np.concatenate(np.delete(paquetsX,i,0))
                    validationT = np.concatenate(np.delete(paquetst,i,0))
                
                    self.entrainement(validationX,validationT)
                    
                    prediction = self.prediction(testX)
                    
                    meanError += self.erreur(testT,prediction)
                    
                meanError = np.mean(meanError)
                
                if ((bestError == -1)) or (meanError <= bestError):
                    bestError = meanError
                    bestLambda = lambda_test
                    bestLearningRate = learningRate_test
                    
        self.lamb = bestLambda
        self.learningRate = bestLearningRate
        
        print("Le meilleur lambda : ",bestLambda)
        print("Le meilleur learning Rate : ",bestLearningRate)
        
        self.entrainement(data,target)
        
        
        
    def prediction(self,x):
        
        return self.model.predict(x)
    
    
      
    @staticmethod  
    def erreur(t,prediction):
        
        error = 0
        for i in range(len(t)):

            if t[i] != prediction[i]:
                error += 1
            else :
                error += 0
        return error

    
X,y = make_classification(n_samples=10, n_features=10,n_informative=5,n_redundant=5,n_classes=3,random_state=1)
# print(X)
# print(len(X))
# print(y)
# print(len(y))

# test = Perceptron(alpha=0.001,eta0=0.001,penalty="l2")
# print(test.loss_function_)
# print(test.loss_functions)
# ovr = OneVsRestClassifier(test)
# ovr.fit(X,y)
# print(ovr.predict(X))
# print(y)
# print(X)