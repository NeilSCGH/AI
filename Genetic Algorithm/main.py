import numpy as np

## PARAMETERS
min=0
max=50

nbPopulation=300
personSize=15

mutation=4 #%
elite=75#%
totalCrossover=False
mutationElite=True

## PARAMETERS


nbElite=int(nbPopulation*elite/100)
nbNonElite=nbPopulation-nbElite

#random population
population=np.random.uniform(min,max,size=(nbPopulation,personSize))

#generating target
cible=np.random.uniform(min,max,size=(1,personSize))

for epoch in range(1000):
    #computing fitness
    fitness=np.sum(abs(population-cible),axis=1)

    #sorting population
    trieur=fitness.argsort()
    fitness=fitness[trieur]
    population=population[trieur]

    if epoch%200==0:
        print(epoch,"\t", np.around(fitness[0]/nbPopulation,3),"\t", np.around(fitness[nbPopulation-1]/nbPopulation,3))

    population=population[:nbElite] #deleting everything except the elite oof the population

    for i in range(nbNonElite):
        parent1=population[np.random.randint(0,nbElite)]
        parent2=population[np.random.randint(0,nbElite)]

        if totalCrossover:#one or the other
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

#combiner randomly avec un tableau de 1 et de 0, multiplier puis additionner les 2 parties complementaires pour le crossover
