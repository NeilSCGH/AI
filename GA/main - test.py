import numpy as np
from core import *

##Data settings
min=0
max=10
personSize=2
nbPop=500
nbEpoch=500

##Fitness part
cible=np.random.uniform(min,max,size=(1,personSize))#generating random target

def computeFitness(pop):
    sum=[]
    for p in pop:
        sum.append(fitness(p))

    return np.array(sum)

def fitness(p):
    r=(p[0] - 5) + (p[1] - 6)*p[0]
    return abs(r)


##Initializing random population
pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Training
best,worst=evolute(pop,computeFitness,nbEpoch=nbEpoch,verbose=20)

#print(best)
print("\nBest :", best)
print("\nBest fitness:", fitness(best))