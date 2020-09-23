import numpy as np
from core import *

##Data settings
min=0
max=10
personSize=2
nbPop=200
nbEpoch=500

##Fitness part
def computeFitness(pop):
    sum=[]
    for p in pop:
        sum.append(fitness(p))

    return np.array(sum)

def fitness(p):
    r1=2*p[0] -2 -p[1]
    r2=-p[0] +3 -p[1]
    return abs(r1)+abs(r2)


##Initializing random population
pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Training
best, _ = evolute(pop,computeFitness,nbEpoch=nbEpoch,verbose=20)

#print(best)
print("\nBest fitness:", fitness(best))
print("Best :", best)