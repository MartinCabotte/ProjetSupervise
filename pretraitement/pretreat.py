import pandas as pd
import numpy as np

def target(data:pd.DataFrame) -> np.array:
    """Fonction permettant de récupérer la liste des "cibles" pour un jeu de données

    Args:
        data (pd.DataFrame): donnee dont on veut les "cibles"

    Returns:
        np.array(classification): tableaux des classes pour chaque element
    """
    
    
    #we get all species name and the number of species

    list_species = []
    for species in data["species"]:
        if species not in list_species:
            list_species.append(species)
            
    #we create the target list
    classification = []
    for element in data["species"]:
        num = list_species.index(element)
        classification.append(num)
 
    return np.array(classification)


            
def pretreat_data(data_train:pd.DataFrame,data_test:pd.DataFrame):
    """_summary_

    Args:
        data_train (pd.DataFrame): jeu de donnee d'entrainement à nettoyer
        data_test (pd.DataFrame): jeu de donnee test à nettoyer

    Returns:
        out: jeu de donnee d'entrainement et jeu de donnee test traite et convertit en "array"
    """
    
    del data_train["id"]
    del data_train["species"]
    del data_test["id"]
    
    data_train_out = []
    data_test_out = []
    
    keys = data_train.keys()
    
    for i in range(len(data_train)):
        temp =[]
        for key in keys : 
            temp.append(data_train[key][i])
        data_train_out.append(temp)
    
    keys = data_test.keys()
    for i in range(len(data_test)):
        temp = []
        for key in keys:
            temp.append(data_test[key][i])
        data_test_out.append(temp)
    
    out = [np.array(data_train_out),np.array(data_test_out)]
    return out



if __name__ == "__main__":
    pretreat_data()