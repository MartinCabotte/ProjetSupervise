import pandas as pd

def pretreat(data):
    
    #we get all species name and the number of species

    list_species = []
    for species in dataset["species"]:
        if species not in list_species:
            list_species.append(species)
            
    #we create the target matrix for data
    
    classification = []        
    counter = 0
    for species in list_species:
        classification.append([])
        for element in dataset["species"]:
            if element == species:
                classification[counter].append(1)
            else:
                classification[counter].append(0)
        counter += 1
    
    return classification
            
    
    
    
if __name__ == "__main__":
    pretreat()