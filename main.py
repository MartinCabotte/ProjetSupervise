from methodes.Random_Forest import Random_ForestClassifier

from methodes.Ridge_Classifier import Ridge_Classifier

from methodes.perceptron import PerceptronClassifier

from methodes.SVM import SVMClassifier


import pretraitement.pretreat as pretreat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    
    os.system("clear")
    data_train = pd.read_csv("train.csv",sep=",",decimal=".")
    data_test = pd.read_csv("test.csv",sep=",",decimal=".")
    
    target= pretreat.target(data_train)
    
    data = pretreat.pretreat_data(data_train,data_test)
    
    
    # data_train = data[0]
    # data_test = data[1]
    
    #On récupère les 770 premières données train pour entrainer le modele et les 220 dernières pour mesure l'efficacité en les envoyant en temps que données test
    #Cela nous permet d'avoir un jeu de données test disposant de "target" 
    
    data_train = data[0][:770]
    data_test = data[0][770:]
    
    target_train = target[:770]
    target_test = target[770:]
    
    launch = True
    print("Bienvenue dans l'analyse du jeu de données par 6 systèmes supervisé différents du groupe Cabotte Martin, Charmoille Maxime et Ducrocq Adrien : \n\n")
    print("Veuillez choisir la méthode que vous souhaitez utiliser : \n")
    print("1 - Perceptron")
    print("2 - Méthodes à noyaux")
    print("3 - Random Forest")
    print("4 - Regression logistique")
    print("5 - En attente")
    print("6 - En attente")
    print("7 - Quitter")
    
    choice = input()
    
    while launch :
        # os.system("clear")
        while choice not in ["1","2","3","4","5","6","7"]:
            print("Veuillez choisir la méthode que vous souhaitez utiliser : \n")
            print("1 - Perceptron")
            print("2 - Méthodes à noyaux")
            print("3 - Random Forest")
            print("4 - Regression logistique")
            print("5 - En attente")
            print("6 - En attente")
            print("7 - Quitter")
            choice = input()
            

        if choice == "1":
            
            # per.entrainement(data_train,target_train)
       
            allPredictions = []
            for i in range(1):
                
                per = PerceptronClassifier(1)
                per.validation_croisee(data_train,target_train)
                # RL.entrainement(data_train,target_train)
                prediction = per.prediction(data_test)
                toSave = prediction.tolist()
                toSave.append(per.erreur_finale(toSave,target_test))
                toSave.append(per.learningRate)
                toSave.append(per.lamb)

                allPredictions.append(toSave)
                
                print(prediction)
                print("l'erreur est de : ", per.erreur_finale(prediction,target_test),"%")
            allPredictions = pd.DataFrame(allPredictions)
            
            allPredictions.to_csv("results/PerceptronOVO.csv")
            print("\n\nEntrez n'importe quelle touche pour revenir au menu principal")
            input()
            choice = "0"
        
        elif choice == "2":
            print("SVM")
            choice_1 = ""
            choice_1_validation = ["1","2","3","4"]
            while choice_1 not in choice_1_validation :
                os.system("clear")
                print("Veuillez choisir le type de noyau : \n")
                print("1 -> noyau RBF")
                print("2 -> noyau polynomial")
                print("3 -> noyau sigmoidale")
                print("4 -> noyau lineaire")
                choice_1 = input()
                
            if choice_1 == "1":
                
                svm = SVMClassifier("rbf")
                
            elif choice_1 == "2":
                
                svm = SVMClassifier("poly")
                
            elif choice_1 == "3":
                
                svm = SVMClassifier("sigmoid")
                
            else:
                
                svm = SVMClassifier("linear")
                
                
            allPredictions = []
            for i in range(10):
                
                svm.validation_croisee(data_train,target_train)
                # RL.entrainement(data_train,target_train)
                prediction = svm.prediction(data_test)
                toSave = prediction.tolist()
                toSave.append(svm.erreur_finale(toSave,target_test))
                toSave.append(svm.gamma)
                toSave.append(svm.coef0)
                toSave.append(svm.M)

                allPredictions.append(toSave)
                
                print(prediction)
                print("l'erreur est de : ", svm.erreur_finale(prediction,target_test),"%")
            allPredictions = pd.DataFrame(allPredictions)
            
            allPredictions.to_csv("results/SVM.csv")
            print("\n\nEntrez n'importe quelle touche pour revenir au menu principal")
            input()
            
            choice = "0"
        
        elif choice == "3":
            allPredictions = []
            for i in range(10):
                
                RD = Random_ForestClassifier()
                RD.validation_croisee(data_train,target_train)
                
                prediction = RD.prediction(data_test)
                toSave = prediction.tolist()
                toSave.append(RD.erreur_finale(toSave,target_test))
                toSave.append(RD.n_estimer)

                allPredictions.append(toSave)
                
                print(prediction)
                print("l'accuracy est de : ", RD.erreur_finale(prediction,target_test),"%")
            allPredictions = pd.DataFrame(allPredictions)
            
            allPredictions.to_csv("results/RandomForestLabels.csv")
            
            print("\n\nEntrez n'importe quelle touche pour revenir au menu principal")
            input()
            choice = "0"
            
        elif choice == "4":
            allPredictions = []
            for i in range(10):
                
                RC = Ridge_Classifier()
                RC.validation_croisee(data_train,target_train)
                # RL.entrainement(data_train,target_train)
                prediction = RC.prediction(data_test)
                toSave = prediction.tolist()
                toSave.append(RC.erreur_finale(toSave,target_test))
                toSave.append(RC.lamb)
                toSave.append(RC.solv)

                allPredictions.append(toSave)
                
                print(prediction)
                print("l'accuracy est de : ", RC.erreur_finale(prediction,target_test),"%")
            allPredictions = pd.DataFrame(allPredictions)
            
            allPredictions.to_csv("results/Ridge_Classifieur.csv")
            print("\n\nEntrez n'importe quelle touche pour revenir au menu principal")
            input()
            choice = "0"
            
        elif choice == "5":
            print("En développement")
            choice = "0"
            
        elif choice == "6":
            print("En développement")
            choice = "0"
        
        else:
            launch = False
    
if __name__ == "__main__":
    main()