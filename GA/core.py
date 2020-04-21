import numpy as np

##Training
def evolute(population,fitnessfunction,nbEpoch,mutation=0.6,elite=5,floatingCrossover=True,mutationElite=False,verbose=0):
    print("ep {}, mut {}, el {}, cross {}, mutel {}".format(nbEpoch,mutation,elite,floatingCrossover,mutationElite))
    nbPopulation,personSize=population.shape

    nbElite=int(nbPopulation*elite/100)
    nbNonElite=nbPopulation-nbElite
    oldMin=0
    for epoch in range(nbEpoch):
        #computing fitness
        fitness=fitnessfunction(population)

        #sorting population
        trieur=fitness.argsort()
        fitness=fitness[trieur]
        population=population[trieur]

        if verbose!=0 and epoch%(nbEpoch/100*verbose)==0:
            pourcent=int(epoch/nbEpoch*100)
            minFitness=np.round(fitness[0],2)
            maxFitness=np.round(fitness[-1],2)
            diff=np.round(maxFitness-minFitness,2)

            if epoch==0:
                print("{}%\t {}-{}\t({}D)".format(pourcent,minFitness,maxFitness,diff))
            else:
                print("{}%\t {}-{}\t({}D) ({}V)".format(pourcent,minFitness,maxFitness,diff,np.round(oldMin-minFitness,2)))

            oldMin=minFitness

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

    fitness=fitnessfunction(population)
    p=population[fitness.argsort()]
    return p[0],p[-1]