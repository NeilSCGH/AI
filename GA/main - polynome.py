import numpy as np
from matplotlib import pyplot as plt
from core import *

##Data settings

points=[[1,2],
        [3,4],
        [5,2],
        [4,1],
        [1.75,4]]

points=[[0,0],
        [1,1],
        [2,2],
        [3,3]]


max=100
nbPop=500
nbEpoch=2000

min=-max
personSize=len(points)
##Fitness part
def computeFitness(pop):
    sum=[]
    for p in pop:
        sum.append(fitness(p))

    return np.array(sum)

def fitness(p):
    error=0
    for x,y in points:
        error += (y - np.polyval(p,x))**2
    return error

def PolyCoefficients(x, coeffs):
    list=[]
    for val in x:
        list.append(np.polyval(coeffs,val))
    return list

##Initializing random population
try:
    #assert False #Force new population
    _=pop[0]
    assert len(pop) == nbPop
    assert len(pop[0]) == personSize
    print("USING EXISTING POPULATION !\n")
except:
    print("Initializing population")
    pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Training
pop = evolute(pop,computeFitness,nbEpoch=nbEpoch,verbose=5)
best=pop[0]

#print(best)
print("\nBest fitness:", fitness(best))
print("Best :", best)





#Plotting points
a, b = np.array(points).T
plt.scatter(a,b)

#Plotting polynom
x=np.linspace(np.min(a)-2, np.max(a)+2, 50)
#coeffs = [1, 2, 3]
plt.plot(x, PolyCoefficients(x, best))
plt.ylim(np.min(b)-2, np.max(b)+2)
plt.show()