import numpy as np
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier


class Fusion:
    
    def __init__(self):
        Adaboost = pd.read_csv("results/AdaboostLabels.csv",sep=",",decimal=".")
        Adaboost = Adaboost.loc[0, ~Adaboost.columns.str.contains('^Unnamed')]
        Adaboost = Adaboost[0:220] #Get 220 test values

        PerceptronOVO = pd.read_csv("results/PerceptronOVO.csv",sep=",",decimal=".")
        PerceptronOVO = PerceptronOVO.loc[0, ~PerceptronOVO.columns.str.contains('^Unnamed')]
        PerceptronOVO = PerceptronOVO[0:220] #Get 220 test values

        PerceptronOVR = pd.read_csv("results/PerceptronOVR.csv",sep=",",decimal=".")
        PerceptronOVR = PerceptronOVR.loc[7, ~PerceptronOVR.columns.str.contains('^Unnamed')]
        PerceptronOVR = PerceptronOVR[0:220] #Get 220 test values

        RandomForest = pd.read_csv("results/RandomForestLabels.csv",sep=",",decimal=".")
        RandomForest = RandomForest.loc[5, ~RandomForest.columns.str.contains('^Unnamed')]
        RandomForest = RandomForest[0:220] #Get 220 test values

        Ridge = pd.read_csv("results/Ridge_Classifieur.csv",sep=",",decimal=".")
        Ridge = Ridge.loc[2, ~Ridge.columns.str.contains('^Unnamed')]
        Ridge = Ridge[0:220] #Get 220 test values

        SVM2 = pd.read_csv("results/SVM2.csv",sep=",",decimal=".")
        SVM2 = SVM2.loc[1, ~SVM2.columns.str.contains('^Unnamed')]
        SVM2 = SVM2[0:220] #Get 220 test values

        SVM4 = pd.read_csv("results/SVM4.csv",sep=",",decimal=".")
        SVM4 = SVM4.loc[7, ~SVM4.columns.str.contains('^Unnamed')]
        SVM4 = SVM4[0:220] #Get 220 test values

        
        majorityVote = []
        for i in range(220):
            ithValue = np.array([Adaboost[i],PerceptronOVO[i],PerceptronOVR[i],RandomForest[i],Ridge[i],SVM2[i],SVM4[i]])
            uniqueithValue = np.unique(ithValue)
            countithValue = []
            for j in range(len(uniqueithValue)):
                countithValue.append(np.count_nonzero(ithValue == uniqueithValue[j]))
            majorityVote.append(uniqueithValue[np.argmax(countithValue)])

        self.labels = majorityVote
    
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