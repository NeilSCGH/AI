import numpy as np
from matplotlib import pyplot as plt
from core import *

##Data settings

points=[[4.294, 100],
        [4.236, 100],
        [4.224, 99],
        [4.145, 93],
        [4.045, 88],
        [4.011, 84],
        [4.212, 99],
        [4.246, 97],
        [4.178, 93],
        [4.18, 92],
        [4.138, 88],
        [4.038, 81],
        [4.023, 79],
        [4.024, 78],
        [3.941, 75],
        [3.844, 63],
        [3.854, 61],
        [3.808, 59],
        [3.794, 57],
        [3.802, 56],
        [3.793, 49],
        [3.76, 48],
        [3.795, 48],
        [3.735, 42],
        [3.722, 37],
        [3.7, 25]]

max=10
min=0

nbPop=2000

personSize=4

def f(p, x):
    a=p[0]
    b=p[1]
    c=p[2]
    d=p[3]
    return (-a * np.exp(-b*x + 10*c) + d)*100

##Fitness part
def computeFitness(pop):
    sum=[]
    for p in pop:
        sum.append(fitness(p))

    return np.array(sum)

def fitness(p):
    error=0
    for x,y in points:
        error += (y - f(p, x))**2
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
pop = evolute(pop,computeFitness,mutation=0.1,mutationElite=False,elite=0.5,nbEpoch=50,verbose=1,min=min,max=max)
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