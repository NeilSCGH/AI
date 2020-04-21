import numpy as np
import matplotlib.pyplot as plt
from functions import *
import sys

sys.path.append('..\GA')
from core import *

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

############# LEARNING #############
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



##Training
best,worst=evolute(pop,computeFitness,nbEpoch=10,verbose=20)

#print(best)
print("\nBest fitness:",computeFitness(best)[0])
print("Worst fitness:",computeFitness(worst)[0])





























assert False
for epoch in range(1,nbEpoch+1):
    # Forward Propagation
    Yp,F,Fb,Xb = fwp(X,V,W)

    #Computing Error
    E = error(Y,Yp,J)

    #Printing Graph
    if showGraph and epoch % graphEpoch==0:
        xAxis.append(epoch)
        EGraph.append(E)

        plt.plot(xAxis,EGraph)
        fig.canvas.draw()

    #Printing error
    if epoch % printEpoch==0:
        print("epoch", epoch, ":", "%.3f" % E)


    #BACK Propagation
    V,W = bp(V,W,Y,Yp,F,Fb,Xb,J,K,N,av,aw)

    #Change ac and aw
    av *= aEvolution
    aw *= aEvolution

if showGraph: plt.show()

print()
print()

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



