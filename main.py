from perceptron import PerceptronClassifier
import pretreat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    data_train = pd.read_csv("train.csv",sep=",",decimal=".")
    data_test = pd.read_csv("test.csv",sep=",",decimal=".")
    
    target_train = pretreat.target(data_train)
    
    data_train,data_test = pretreat.pretreat_data(data_train,data_test)
    
    
    launch = True
    print("Bienvenue dans l'analyse du jeu de données par 6 systèmes supervisé différents du groupe Cabotte Martin, Charmoille Maxime et Ducrocq Adrien : \n\n")
    print("Veuillez choisir la méthode que vous souhaitez utiliser : \n")
    print("1 - Perceptron")
    print("2 - Méthodes à noyaux")
    print("3 - En attente")
    print("4 - En attente")
    print("5 - En attente")
    print("6 - En attente")
    print("7 - Quitter")
    
    choice = input()
    
    while launch :
        
        while choice not in ["1","2","3","4","5","6","7"]:
            print("Veuillez choisir la méthode que vous souhaitez utiliser : \n")
            print("1 - Perceptron")
            print("2 - En attente")
            print("3 - En attente")
            print("4 - En attente")
            print("5 - En attente")
            print("6 - En attente")
            print("7 - Quitter")
            choice = input()
            

        if choice == "1":
            per = PerceptronClassifier(1,0)
            # per.entrainement(data_train,target_train)
            per.validation_croisee(data_train,target_train)
            print(per.prediction(data_test))
            choice = "0"
        
        elif choice == "2":
            print("En développement")
            choice = "0"
        
        elif choice == "3":
            print("En développement")
            choice = "0"
            
        elif choice == "4":
            print("En développement")
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