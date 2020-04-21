import numpy as np
import matplotlib.pyplot as plt
from functions import *
import sys

sys.path.append('..\GA')
from core import *


def extractWeights(person,VSize,WSize,N,K,J):
    person=person[0]

    V=person[:VSize]
    W=person[VSize:]

    V=V.reshape((N+1,K))#V.shape=(N+1,K)
    W=W.reshape((K+1,J))#W.shape=(K+1,J)

    return V,W

def fitnessPerson(person):
    person=person.reshape((1,63))
    V,W=extractWeights(person,VSize,WSize,N,K,J)

    #forward propagation
    Yp,F,Fb,Xb = fwp(X,V,W)

    #Computing Error
    E = error(Y,Yp,J)
    return E

def computeFitness(pop):
    result=[]
    for pers in pop:
        #print(pers)
        result.append(fitnessPerson(pers))

    return np.asarray(result)

############# SETTINGS #############
#NN
K=10

#GA
nbPop=200

############# INITIALIZING #############
#Getting X and Y
X, Y, YUnique = getData("data.txt")

N=len(X[0])
J=len(Y[0])

VSize=(N+1)*K
WSize=(K+1)*J
personSize= VSize+WSize

pop=np.random.uniform(-1,1,size=(nbPop,personSize))


##Training
best,worst=evolute(pop,computeFitness,nbEpoch=100,verbose=10,mutation=30,elite=1)

#print(best)
print("\nBest fitness:",fitnessPerson(best))
#Oprint("Worst fitness:",fitnessPerson(worst))
print("")


V,W=extractWeights(best.reshape((1,63)),VSize,WSize,N,K,J)
Yp,F,Fb,Xb = fwp(X,V,W)
##Printing results
# print(Y)
# print(np.apply_along_axis(arrondi, 0, Yp))
# print()
# print()

##Testing
XTest=[[2,2],[4,4],[4.5,1.5],[1.5,1]]

R=fwp(XTest,V,W)

R=R[0]
R=np.apply_along_axis(arrondi, 0, R)

for i in range(len(XTest)):
    rCateg=R[i]
    r=sum(YUnique*rCateg)
    print(XTest[i], " \t", rCateg, " \t", r)



