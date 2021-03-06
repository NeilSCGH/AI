import numpy as np
from core import *

##Data settings
min=0
max=1000
personSize=10
nbPop=500

##Fitness part
cible=np.random.uniform(min,max,size=(1,personSize))#generating random target

def computeFitness(pop):
    return np.sum(abs(pop-cible),axis=1)


##Initializing random population
pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Training
nbEpoch=500
best,worst=evolute(pop,computeFitness,nbEpoch=nbEpoch,verbose=20)

#print(best)
print("\nBest fitness:",computeFitness(best)[0])
print("Worst fitness:",computeFitness(worst)[0])