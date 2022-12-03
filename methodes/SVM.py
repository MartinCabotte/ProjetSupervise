import pandas as pd
import numpy as np
import os
from sklearn.svm import NuSVC
class SVMClassifier:
    
    def __init__(self,choice:str):
        
        self.kernel = choice
        self.model = ""
        self.nu = 0.5 #utile pour tous (va de 0 à 1)
        self.M = 3 #utile uniquement pour polynomial (degre du polynome)
        self.coef0 = 0 #Utile uniquement pour sigmoide et polynomial
        self.gamma = 0.1#utile pour rbf, poly et sigmoid (par défaut 1/n)
        self.method = "ovr"
        
        
    def entrainement(self,data_train:np.array,target_train:np.array):
        """Fonction permettant d'entrainer le modele

        Args:
            data_train (np.array): donnee d'entrainement
            target_train (np.array): cible correspondante pour les donnees d'entrainement
        """
        svm = NuSVC(nu=self.nu,kernel=self.kernel,degree=self.M,coef0=self.coef0,decision_function_shape=self.method,gamma=self.gamma)
        self.model = svm.fit(data_train,target_train)
        
        
        

    def validation_croisee(self,data:np.array,target:np.array):
        """Fonction permettant la validation croisee afin de trouver les meilleurs hyperparametres

        Args:
            data (np.array): donnee d'entrainement
            target (np.array): cible correspondante aux donnees d'entrainement
        """
        K = 10
            
        bestError = -1
        bestNu = 0
        bestM = 1
        bestCoeff0 = 0
        bestGamma = 1
        
        #on mélange tout d'abord nos données
        index = np.arange(0,len(data),1)
        np.random.shuffle(index)

        data = data[index]
        target = target[index]


        #On les divise en k paquets
        paquetsX = np.array_split(data,K)
        paquetst = np.array_split(target,K)
        
        
        if self.kernel == "linear" :
            
            #on réalise les simulations
            for nu_test in np.arange(0.001,0.3,0.001):
                self.nu = nu_test
                print(self.nu)
                meanError = 0
                
                for i in range(0,K):
                        
                    testX = paquetsX[i]
                    testT = paquetst[i]
                    
                    validationX = np.concatenate(np.delete(paquetsX,i,0))
                    validationT = np.concatenate(np.delete(paquetst,i,0))
                
                    self.entrainement(validationX,validationT)
                    
                    prediction = self.prediction(testX)
                    
                    meanError += self.erreur(testT,prediction,1)
                        
                meanError = np.mean(meanError)
                
                if ((bestError == -1)) or (meanError <= bestError):
                    bestError = meanError
                    bestNu = nu_test

            
        elif self.kernel == "rbf":
            
            #on réalise les simulations
            for nu_test in np.arange(0.01,0.3,0.01):
                self.nu = nu_test
                print("nu = ",self.nu)
                
                meanError = 0
                
                for gamma_test in np.arange(0.01,1,0.01):
                    
                    self.gamma = gamma_test
                    print("gamma = ",gamma_test)
                    
                    for i in range(0,K):
                        
                        testX = paquetsX[i]
                        testT = paquetst[i]
                        
                        validationX = np.concatenate(np.delete(paquetsX,i,0))
                        validationT = np.concatenate(np.delete(paquetst,i,0))
                    
                        self.entrainement(validationX,validationT)
                        
                        prediction = self.prediction(testX)
                        
                        meanError += self.erreur(testT,prediction,1)
                        
                    meanError = np.mean(meanError)
                    
                    if ((bestError == -1)) or (meanError <= bestError):
                        bestError = meanError
                        bestNu = nu_test
                        bestGamma = gamma_test
            
            
        elif self.kernel == "poly":
            
            #on réalise les simulations
            for nu_test in np.arange(0.01,0.3,0.01):
                self.nu = nu_test
                print("nu = ",self.nu)
                
                meanError = 0
                
                for gamma_test in np.arange(0.01,1,0.01):
                    
                    self.gamma = gamma_test
                    print("gamma = ",gamma_test)
                    
                    for M_test in range(1,10):
                        self.M = M_test
                        print("M = ",M_test)
                        
                        for coef0_test in np.arange(0,10,1):
                            self.coef0 = coef0_test
                            print("coef0 = ",coef0_test)
                            
                            
                            for i in range(0,K):
                                
                                testX = paquetsX[i]
                                testT = paquetst[i]
                                
                                validationX = np.concatenate(np.delete(paquetsX,i,0))
                                validationT = np.concatenate(np.delete(paquetst,i,0))
                            
                                self.entrainement(validationX,validationT)
                                
                                prediction = self.prediction(testX)
                                
                                meanError += self.erreur(testT,prediction,1)
                                
                            meanError = np.mean(meanError)
                            
                            if ((bestError == -1)) or (meanError <= bestError):
                                bestError = meanError
                                bestNu = nu_test
                                bestGamma = gamma_test
                        
                        
                        
        elif self.kernel == "sigmoid":
            
            #on réalise les simulations
            for nu_test in np.arange(0.1,0.3,0.1):
                self.nu = nu_test
                print("nu = ",self.nu)
                
                meanError = 0
                
                for gamma_test in np.arange(0.1,0.3,0.1):
                    
                    self.gamma = gamma_test
                    print("gamma = ",gamma_test)
                    
                        
                    for coef0_test in np.arange(-1,5,1):
                        self.coef0 = coef0_test
                        print("coef0 = ",coef0_test)
                        
                        
                        for i in range(0,K):
                            
                            testX = paquetsX[i]
                            testT = paquetst[i]
                            
                            validationX = np.concatenate(np.delete(paquetsX,i,0))
                            validationT = np.concatenate(np.delete(paquetst,i,0))
                        
                            self.entrainement(validationX,validationT)
                            
                            prediction = self.prediction(testX)
                            
                            meanError += self.erreur(testT,prediction,1)
                            
                        meanError = np.mean(meanError)
                        
                        if ((bestError == -1)) or (meanError <= bestError):
                            bestError = meanError
                            bestNu = nu_test
                            bestGamma = gamma_test
        
        
        print(bestNu)
        self.nu = bestNu
        self.coef0 = bestCoeff0
        self.M = bestM
        self.gamma = bestGamma
        os.system("clear")
        
        if self.kernel == "linear" :
            
            print("Le meilleur nu : ",bestNu)
            
        elif self.kernel == "rbf":
            
            print("Le meilleur nu : ",bestNu)
            print("Le meilleur gamma : ",bestGamma)
            
        elif self.kernel == "poly":
            
            print("Le meilleur nu : ",bestNu)
            print("Le meilleur gamma : ",bestGamma)
            print("Le meilleur M : ",bestM)
            print("Le meilleur coeff0 : ",bestCoeff0)
            
        elif self.kernel == "sigmoid":
            
            print("Le meilleur nu : ",bestNu)
            print("Le meilleur gamma : ",bestGamma)
            print("Le meilleur coeff0 : ",bestCoeff0)
            
        
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
    def erreur(t:np.array,prediction:np.array,choix:int) -> int:
        """fonction retournant l'erreur de prediction lors de l'entrainement du modele pour une valeur donnee

        Args:
            t (np.array): liste contenant le numero de la classe à laquelle appartient chaque element
            prediction (np.array): liste contenant le numero de la classe à laquelle appartient chaque selon le modele
            choix (int) : choix de l'erreur : 0 = quadratique, 1 = svm loss
        Returns:
            error (int) : l'erreur du modele
        """
        error = 0
        for i in range(len(t)) :
            if choix == 0:
                error += (t[i] - prediction[i])**2 
            else:
                error += max(0,1-t[i]*prediction[i])
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