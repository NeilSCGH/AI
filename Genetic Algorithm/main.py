import numpy as np
from core import *

##Data settings
min=0
max=1000
personSize=10
nbPop=500

##Initialization
#random population
pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Fitness part
#generating target
cible=np.random.uniform(min,max,size=(1,personSize))

def computeFitness(pop):
    return np.sum(abs(pop-cible),axis=1)

##Training
best,worst=evolute(pop,computeFitness,nbEpoch=500,,verbose=20)

#print(best)
print("\nBest fitness:",computeFitness(best)[0])
print("Worst fitness:",computeFitness(worst)[0])