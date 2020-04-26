import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('..\GA')
from core import *

def fitnessPers(pers):
    tmp=X*pers
    tmp=np.sum(tmp,axis=1)
    tmp=np.abs(tmp)
    tmp=np.sum(tmp)

    return int(tmp)

def computeFitness(pop):
    result=[]
    for pers in pop:#for each person in the population
        result.append(fitnessPers(pers))

    return np.asarray(result)

############# SETTINGS #############
#GA
nbPop=500

############# INITIALIZING #############
df_clean = pd.read_csv('data_cleaned.csv')
#Getting X and Y
targetColumn='Overall'
Y = df_clean[targetColumn] #result column
XRaw = df_clean.drop(targetColumn, axis=1, inplace=False)

N,personSize=XRaw.shape
columns=list(XRaw)
X=[]
for i in range(N):#for each line
    X.append(list(XRaw.loc[i]))
X=np.asarray(X)


pop=np.random.uniform(-1,1,size=(nbPop,personSize))


##Training
pop,best,worst=evolute(pop,computeFitness,nbEpoch=20,verbose=10,mutation=10,elite=50,min=-1,max=1)


##Printing results
#print(best)
# print("\nBest fitness:",fitnessPerson(best))
#Oprint("Worst fitness:",fitnessPerson(worst))
# print("")

for i in range(personSize):
    print("{} \t{} \t\t{}".format(np.round(best[i],2),XRaw.columns[i], X[0][i]))

np.set_printoptions(suppress=True)
#scores=np.round(computeFitness(pop))
#print(scores)

print("\nTOP 5")
for i in range(0,5):
    pers=pop[i]
    print("{}:{}".format(i+1,np.round(fitnessPers(pers),3)))


def testPerson(pers,idSet):
    print("{}: {} vs {}".format(idSet,np.sum(X[idSet]*pers),Y[idSet]))


testPerson(best,0)
testPerson(best,10)