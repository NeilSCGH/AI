import numpy as np

##Data settings
min=0
max=1000
personSize=10
nbPop=500

##Initialization
#random population
pop=np.random.uniform(min,max,size=(nbPop,personSize))

##Fitnes part
#generating target
cible=np.random.uniform(min,max,size=(1,personSize))

def computeFitness(pop):
    return np.sum(abs(pop-cible),axis=1)

##Training
def evolute(population,nbEpoch=5000,mutation=0.6,elite=5,floatingCrossover=False,mutationElite=True,verbose=0):
    print("ep {}, mut {}, el {}, cross {}, mutel {}\n".format(nbEpoch,mutation,elite,floatingCrossover,mutationElite))
    nbPopulation=len(pop)

    nbElite=int(nbPopulation*elite/100)
    nbNonElite=nbPopulation-nbElite

    for epoch in range(nbEpoch):
        #computing fitness
        fitness=computeFitness(population)

        #sorting population
        trieur=fitness.argsort()
        fitness=fitness[trieur]
        population=population[trieur]

        if verbose!=0 and epoch%(nbEpoch/100*verbose)==0:
            pourcent=int(epoch/nbEpoch*100)
            minFitness=int(np.round(fitness[0]))
            maxFitness=int(np.round(fitness[-1]))
            diff=maxFitness-minFitness
            print("{}%\t {}-{} ({})".format(pourcent,minFitness,maxFitness,diff))

        population=population[:nbElite] #deleting everything except the elite oof the population

        for i in range(nbNonElite):
            parent1=population[np.random.randint(0,nbElite)]
            parent2=population[np.random.randint(0,nbElite)]

            if floatingCrossover:#one or the other
                cm=np.random.randint(0,1,size=(1,personSize)) #generating the crossover mask
            else:#a mix of both
                cm=np.random.uniform(0,1,size=(1,personSize)) #generating the crossover mask

            leNouveau = parent1*cm + parent2*(1-cm) #generating the new person
            population = np.concatenate((population,leNouveau)) #adding it to the population

        if mutationElite:
            muteur=np.random.uniform(1-mutation/100,1+mutation/100,size=(nbPopulation,personSize))
        else:
            muteur=np.random.uniform(1-mutation/100,1+mutation/100,size=(nbNonElite,personSize))
            ones=np.ones((nbElite,personSize))
            muteur=np.concatenate((ones,muteur))

        population=population*muteur
        #population = population.astype(int)

    fitness=computeFitness(population)
    p=population[fitness.argsort()]
    return p[0],p[-1]


best,worst=evolute(pop,nbEpoch=5000,mutation=0.6,elite=5,floatingCrossover=True,mutationElite=False,verbose=5)
#print(best)
print("\nfitness:",computeFitness(best)[0])
print("fitness:",computeFitness(worst)[0])