import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:
    
    def __init__(self):
        """on initie la classe Adaboost"""
        
        self.n_estimer = 10
        self.learningRate = 0.01
        self.model = ""
        
    def entrainement(self,data_train:np.array,target_train:np.array):
        """Fonction permettant d'entrainer le modele

        Args:
            data_train (np.array): donnee d'entrainement
            target_train (np.array): cible correspondante pour les donnees d'entrainement
        """
        Adaboost=AdaBoostClassifier(n_estimators = self.n_estimer, learning_rate = self.learningRate)
        
        self.model = Adaboost.fit(data_train,target_train)
        
        
        

    def validation_croisee(self,data:np.array,target:np.array):
        """Fonction permettant la validation croisee afin de trouver les meilleurs hyperparametres

        Args:
            data (np.array): donnee d'entrainement
            target (np.array): cible correspondante aux donnees d'entrainement
        """
        K = 10
            
        bestError = -1
        bestn_estimer = 0
        bestlr = 0
        
        #on mélange tout d'abord nos données
        index = np.arange(0,len(data),1)
        np.random.shuffle(index)

        data = data[index]
        target = target[index]


        #On les divise en k paquets
        paquetsX = np.array_split(data,K)
        paquetst = np.array_split(target,K)
        
        
        #on réalise les simulations
        for n_estimer_test in np.arange(50,250,50):
            self.n_estimer = n_estimer_test
            print(self.n_estimer)

            for lr_test in np.arange(0.01,0.1,0.01):
                self.learningRate = lr_test
            
                meanError = 0
            
                for i in range(0,K):
                    
                    testX = paquetsX[i]
                    testT = paquetst[i]
                    
                    validationX = np.concatenate(np.delete(paquetsX,i,0))
                    validationT = np.concatenate(np.delete(paquetst,i,0))
                
                    self.entrainement(validationX,validationT)
                    
                    prediction = self.prediction(testX)
                    
                    meanError += self.erreur(testT,prediction,testX)
                    
                
                meanError = np.mean(meanError)
                print(meanError)
                
                
                if ((bestError == -1)) or (meanError <= bestError):
                    bestError = meanError
                    bestn_estimer = n_estimer_test
                    bestlr = lr_test
                
        self.n_estimer = bestn_estimer
        self.learningRate = bestlr
        # os.system("clear")
        print("Le meilleur n estimators est : ",self.n_estimer)
        print("Le meilleur learning rate est : ",self.learningRate)
        
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
    def erreur(t:np.array,prediction:np.array,data_entrainement:list) -> int:
        """fonction retournant l'erreur de prediction lors de l'entrainement du modele pour une valeur donnee

        Args:
            t (np.array): liste contenant le numero de la classe à laquelle appartient chaque element
            prediction (np.array): liste contenant le numero de la classe à laquelle appartient chaque selon le modele
            methode (int): nombre permettant de choisir l'erreur a appliquer
        Returns:
            error (int) : l'erreur du modele
        """
        error = 0
        for i in range(len(t)):
            if t[i] != prediction[i]:
                error += 1
            
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
        