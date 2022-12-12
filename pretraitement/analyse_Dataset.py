import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import dataset as dt

def main():
    dataset = pd.read_csv("train.csv",sep=",",decimal=".")

    #Overview of the dataset
    print(dataset)

    #Count iteration of labels in species column
    countLabel = dataset.groupby(["species"]).size()
    print(countLabel)

    #Compute several stats of the dataset
    print("Nombre d'occurences par label : ",countLabel.unique())
    print("Nombre de valeurs négatives dans le dataset : ",dataset.drop(["id","species"],axis=1).lt(0).sum().sum())
    print("Nombre de valeurs supérieures à 1 dans le dataset : ",dataset.drop(["id","species"],axis=1).gt(1).sum().sum())
    print("Moyenne : ",dataset.drop(["id","species"],axis=1).mean().mean())
    print("Nombre de 0 moyen par attributs : ",int(dataset.drop(["id","species"],axis=1).eq(0).sum().mean()))
    median = int(dataset.drop(["id","species"],axis=1).eq(0).sum().quantile(q=0.5))
    Q1 = int(dataset.drop(["id","species"],axis=1).eq(0).sum().quantile(q=0.25))
    Q3 = int(dataset.drop(["id","species"],axis=1).eq(0).sum().quantile(q=0.75))
    print("Médiane, Q1 et Q3 de 0 par attributs : ", median, Q1, Q3)

    #In the previous section, we have seen that datas are between 0 and 1, mean value is 0.01
    #Mean of zeros per attribute is 203
    #Median of number of zeros per attribute is 133, Q1 is 0 and Q3 is 313

    #Now, let's analyze the repartition of zeros per attribute in the dataset.
    #First, let's print a barplot of zeros  in the dataset
    arrayOfZeros = dataset.drop(["id","species"],axis=1).eq(0).sum()
    
    toPlot = np.bincount(np.array(arrayOfZeros))
    plt.plot(range(len(toPlot)),toPlot)
    plt.title("Number of zeros per attributes")
    plt.xlabel("Number of zeros in the attribute")
    plt.ylabel("Number of attributes with this number of 0")
    plt.show()  

    
    plt.plot(range(Q3,len(toPlot)),toPlot[Q3:len(toPlot)])
    plt.title("Number of zeros per attributes from Q3 to the end")
    plt.xlabel("Number of zeros in the attribute")
    plt.ylabel("Number of attributes with this number of 0")
    plt.show() 

    possiblyRepresentative = arrayOfZeros[arrayOfZeros>=(2*Q3)]
    print("Ces attributs sont peut être très représentatifs dans nos données car il y a beaucoup de 0 dans le dataset : ",possiblyRepresentative)

    #La transformation du dataset sera la suivante : arrayOfZeros / max(arrayOfZeros)
    maxZeros = max(arrayOfZeros)
    weightZeros = arrayOfZeros/maxZeros
    datasetToApplyWeights = dataset.drop(["id","species"],axis=1)
    datasetToApplyWeights = datasetToApplyWeights * weightZeros
    datasetToApplyWeights = pd.DataFrame(datasetToApplyWeights)

    datasetToApplyWeights.to_csv("weightedDataset.csv")

    #Maintenant, on va analyser les sorties spectrals de nos algorithmes
    datasets3 = dt.dataset(dataset.drop(["id","species"],axis=1),3)
    datasets7 = dt.dataset(dataset.drop(["id","species"],axis=1),7)

    datasets3.spectral.to_csv("spectralDataset3Neighboor.csv")
    datasets7.spectral.to_csv("spectralDataset7Neighboor.csv")

if __name__ == "__main__":
    main()