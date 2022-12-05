import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier


from sklearn.datasets import make_classification


class PerceptronClassifier:
    
    def __init__(self,mode=0):
        """on initie la classe du perceptron

        Args:
            mode (int, optional): permet de choisir si on réalise une analyse en mode "One VS Rest" (mode = 0) ou en mode "One VS One" (mode = 1).  0 par défaut.
        """
        
        self.lamb = 0.0001
        self.learningRate = 0.01
        self.w = np.array([0,5]) #parametre aléatoire
        self.w0 = -10 #parametre aleatoire
        self.mode = mode #determination du model : 0 "OneVsRest" ; 1 : "OneVsOne"
        self.model = ""
        
    def entrainement(self,data_train:np.array,target_train:np.array):
        """Fonction permettant d'entrainer le modele

        Args:
            data_train (np.array): donnee d'entrainement
            target_train (np.array): cible correspondante pour les donnees d'entrainement
        """
        per = Perceptron(alpha=self.lamb,eta0=self.learningRate,penalty="l2")
        
        if self.mode == 0:
        
            model = OneVsRestClassifier(per)
            
        else : 
            
            model = OneVsOneClassifier(per)
            
        self.model = model.fit(data_train,target_train)
        
        

    def validation_croisee(self,data:np.array,target:np.array):

        """Fonction permettant la validation croisee K fois afin de trouver les meilleurs hyperparametres

        Args:
            data (np.array): donnee d'entrainement
            target (np.array): cible correspondante aux donnees d'entrainement
        """
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
        for lambda_test in np.arange(0.001,0.01,0.001):
            self.lamb = lambda_test
            print(self.lamb)
            
            meanError = 0
            for learningRate_test in np.arange(0.001,0.01,0.001):
                self.learningRate = learningRate_test
                print(learningRate_test)
                
                for i in range(0,K):
                    
                    testX = paquetsX[i]
                    testT = paquetst[i]
                    
                    validationX = np.concatenate(np.delete(paquetsX,i,0))
                    validationT = np.concatenate(np.delete(paquetst,i,0))
                
                    self.entrainement(validationX,validationT)
                    
                    prediction = self.prediction(testX)
                    

                    meanError += self.erreur(testT,prediction,testX,1)
                    
                meanError = np.mean(meanError)
                
                #On met à jour l'erreur la plus basse et les hyperparamètres associées
                if ((bestError == -1)) or (meanError <= bestError):
                    bestError = meanError
                    bestLambda = lambda_test
                    bestLearningRate = learningRate_test
                    
        self.lamb = bestLambda
        self.learningRate = bestLearningRate
        os.system("clear")
        print("Le meilleur lambda : ",bestLambda)
        print("Le meilleur learning Rate : ",bestLearningRate)
        
        #une fois les meilleurs hyperparametres trouves, on réentraine le modele avec le jeu complet de donnees
        self.entrainement(data,target)
        
        
        
    def prediction(self,x:np.array) -> np.array:
        """methode permettant de retourner la prediction du modele

        Args:
            x (np.array): donnee dont on souhaite une prediction

        Returns:
            prediction : prediction du modele
        """
        
        prediction = self.model.predict(x)
        return prediction
    
    
      
    @staticmethod  
    def erreur(t:np.array,prediction:np.array,data_entrainement:list,methode:int) -> int:
        """fonction retournant l'erreur de prediction lors de l'entrainement du modele pour une valeur donnee

        Args:
            t (np.array): liste contenant le numero de la classe à laquelle appartient chaque element
            prediction (np.array): liste contenant le numero de la classe à laquelle appartient chaque selon le modele

            methode (int): nombre permettant de choisir l'erreur a appliquer

        Returns:
            error (int) : l'erreur du modele
        """
        error = 0

        if methode == 0:
            for i in range(len(t)):

                if t[i] != prediction[i]:
                    error += 1
                else :
                    error += 0
        elif methode == 1:
            for i in range(len(t)):
                if t[i] != prediction[i]:
                    error += (-data_entrainement[i]*t[i])
            
        return error


    @staticmethod
    def erreur_finale(prediction:np.array,target_test:np.array):
        """Fonction permettant de retourner l'erreur de test du modele

        Args:
            prediction (np.array): prediction des données test
            target_test (np.array): "cibles" réelles des données test

        Returns:
            _type_: _description_
        """
        error = 0
        for i in range(len(prediction)):
            if prediction[i] != target_test[i]:
                error += 1
        error = error / len(prediction) * 100
        
        return error
        