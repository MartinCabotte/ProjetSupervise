import pandas as pd
import numpy as np

def target(data):
    
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

def target_manual(data):
    
    #we get all species name and the number of species

    list_species = []
    for species in data["species"]:
        if species not in list_species:
            list_species.append(species)
            
    #we create the target matrix for data
    
    classification = []        
    counter = 0
    for species in list_species:
        classification.append([])
        for element in data["species"]:
            if element == species:
                classification[counter].append(1)
            else:
                classification[counter].append(0)
        counter += 1
    
    return np.array(classification)
            
def pretreat_data(data_train,data_test):

    
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
    
    
    return np.array(data_train_out),np.array(data_test_out)



if __name__ == "__main__":
    pretreat_data()