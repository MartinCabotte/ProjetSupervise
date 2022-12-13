from methodes.Fusion import Fusion

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pretraitement.pretreat as pretreat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    dataToLabellize = pd.read_csv("train.csv",sep=",",decimal=".")
    target= pretreat.target(dataToLabellize)
    target_test = target[770:]

    Fs = Fusion()
    FusionLabels = Fs.labels
    FusionError = 100 - Fs.erreur_finale(FusionLabels,target_test)
    

    PerceptronOVOLabels = pd.read_csv("results/PerceptronOVO.csv",sep=",",decimal=".")
    PerceptronOVO = 100 - PerceptronOVOLabels["220"]

    RandomForestLabels = pd.read_csv("results/RandomForestLabels.csv",sep=",",decimal=".")
    RandomForest = RandomForestLabels["220"]

    RidgeLabels = pd.read_csv("results/Ridge_Classifieur.csv",sep=",",decimal=".")
    Ridge = RidgeLabels["220"]

    SVM2Labels = pd.read_csv("results/SVM2.csv",sep=",",decimal=".")
    SVM2 = 100 - SVM2Labels["220"]

    fs = 10
    fig, axs = plt.subplots(ncols=3, figsize=(6, 6), sharey=True)
    axs[0].boxplot(RandomForest, labels="A")
    axs[0].set_title('RandomForest Accuracy Boxplot', fontsize=fs)
    axs[1].boxplot(Ridge, labels="A")
    axs[1].set_title('Ridge Accuracy Boxplot', fontsize=fs)
    axs[2].boxplot(SVM2, labels="A")
    axs[2].set_title('SVM using polynomial kernel Accuracy Boxplot', fontsize=fs)
    fig.subplots_adjust(hspace=0.4)
    plt.show()

    PrecFus = np.mean(precision_score(target_test,FusionLabels, average=None))
    RecFus = np.mean(recall_score(target_test,FusionLabels, average=None))
    ARIFus = adjusted_rand_score(target_test,FusionLabels)

    print("La précision de la fusion est : "+str(PrecFus))
    print("Le recall de la fusion est : "+str(RecFus))
    print("L'ARI de la fusion est : "+str(ARIFus))


    PerceptronOVOLabels = PerceptronOVOLabels.iloc[0][1:221]

    PrecOVO = np.mean(precision_score(target_test,PerceptronOVOLabels, average=None))
    RecOVO = np.mean(recall_score(target_test,PerceptronOVOLabels, average=None))
    ARIOVO = adjusted_rand_score(target_test,PerceptronOVOLabels)

    print("La précision du perceptron est : "+str(PrecOVO))
    print("Le recall du perceptron est : "+str(RecOVO))
    print("L'ARI du perceptron est : "+str(ARIOVO))
    
    PrecRF = 0
    RecRF = 0
    ARIRF = 0

    for i in range(RandomForestLabels.shape[0]):
        tempRow = RandomForestLabels.iloc[i][1:221]
        PrecRF += np.mean(precision_score(target_test,tempRow, average=None))
        RecRF += np.mean(recall_score(target_test,tempRow, average=None))
        ARIRF += adjusted_rand_score(target_test,tempRow)

    PrecRF = PrecRF / RandomForestLabels.shape[0]
    RecRF = RecRF / RandomForestLabels.shape[0]
    ARIRF = ARIRF / RandomForestLabels.shape[0]

    print("La précision du RandomForest est : "+str(PrecRF))
    print("Le recall du RandomForest est : "+str(RecRF))
    print("L'ARI du RandomForest est : "+str(ARIRF))


    PrecRidge = 0
    RecRidge = 0
    ARIRidge = 0

    print(target_test)

    for i in range(RidgeLabels.shape[0]):
        tempRow = RidgeLabels.iloc[i][1:221].apply(lambda x: float(x))
        PrecRidge += np.mean(precision_score(target_test,tempRow, average=None))
        RecRidge += np.mean(recall_score(target_test,tempRow, average=None))
        ARIRidge += adjusted_rand_score(target_test,tempRow)

    PrecRidge = PrecRidge / RidgeLabels.shape[0]
    RecRidge = RecRidge / RidgeLabels.shape[0]
    ARIRidge = ARIRidge / RidgeLabels.shape[0]

    print("La précision de Ridge est : "+str(PrecRidge))
    print("Le recall de Ridge est : "+str(RecRidge))
    print("L'ARI de Ridge est : "+str(ARIRidge))

    PrecSVM = 0
    RecSVM = 0
    ARISVM = 0

    print(target_test)

    for i in range(SVM2Labels.shape[0]):
        tempRow = SVM2Labels.iloc[i][1:221].apply(lambda x: float(x))
        PrecSVM += np.mean(precision_score(target_test,tempRow, average=None))
        RecSVM += np.mean(recall_score(target_test,tempRow, average=None))
        ARISVM += adjusted_rand_score(target_test,tempRow)

    PrecSVM = PrecSVM / SVM2Labels.shape[0]
    RecSVM = RecSVM / SVM2Labels.shape[0]
    ARISVM = ARISVM / SVM2Labels.shape[0]

    print("La précision du SVM est : "+str(PrecSVM))
    print("Le recall du SVM est : "+str(RecSVM))
    print("L'ARI du SVM est : "+str(ARISVM))
    


if __name__ == "__main__":
    main()