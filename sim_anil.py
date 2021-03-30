

import math
import numpy as np



def simAnil(s0, T0, numberOfSteps, Tfunction, neighborFunction, costFunction, criterion, **params):
    sCurrent, T = s0.copy(), T0
    pizda = 1
    for i in range(numberOfSteps):
        pizda = 1
        sProposed = neighborFunction(sCurrent, **params)
        pizda = 1
        costProposed = costFunction(sProposed, **params)
        costCurrent = costFunction(sCurrent, **params)
        pizda = 1
        if costProposed < costCurrent:
            sCurrent = sProposed
            pizda = 1
        else: 
            if criterion(costCurrent, costProposed, T, **params):
                sCurrent = sProposed
                pizda = 1
        print(T, costFunction(sCurrent, **params))
        T = Tfunction(T0, i, **params)
        pizda = 1
    return sCurrent





def fastSimAnil(T0, i, **params):
    return T0/(i+1)


def symmectricTSPNeighbor(nodes, **params):#should we add params here? So if there above we have neighborFunction(sCurrent, **params), but what if we don't provide any? 
    a, b = 0, 1
    nodes1 = nodes.copy()
    while True:
        a = np.random.choice(len(nodes1), 1)[0]
        b = np.random.choice(len(nodes1), 1)[0]
        if a != b:
            nodes1[a], nodes1[b] = nodes1[b], nodes1[a]
            break
    return nodes1


def symmectricTSPCost(nodes, **params):
    nodes1 = nodes.copy()
    pizda = 1
    nodes1.append(nodes1[0])#before sampling we have to sort them
    hm = np.array([[nodes1[i-1] , nodes1[i]] for i in range(1, len(nodes1))])#can leak memory 
    pizda = 1
    hm.sort(axis=1)
    hm = hm.T
    pizda = 1 
    return np.sum(params['adjacencyMatrix'][hm[0], hm[1]])#or maybe a regular sum? 



def metropolisCriterion(costCurrent, costPorposed, T, **params):
    if math.e ** (-( (costPorposed - costCurrent) / T)) >= np.random.uniform(0, 1, size=1)[0]:
        return True
    else:
        return False 






# print(Tfunctions.fastSimAnil(35, 45))
# a = [0, 10, 15, 25]#can be an np.array too. Maybe it would be better
# b = neighborFunctions.symmectricTSP([0, 10, 15, 25])
# print(a, b)

adjMat = np.array([
    [0, 10, 15, 20]
, [10, 0, 35, 25]
, [15, 35, 0, 30]
, [20, 25, 30, 0]
])


a = simAnil([0, 1, 2, 3], 
    100, 
    1000, 
    fastSimAnil, 
    symmectricTSPNeighbor, 
    symmectricTSPCost, 
    metropolisCriterion, 
    adjacencyMatrix = adjMat
)

print(a)

n = 1000
adjMat = np.absolute(np.random.normal(0, 100, n**2)).reshape(n, n)

a = simAnil(list(range(n)), 
    1000, 
    100000, 
    fastSimAnil, 
    symmectricTSPNeighbor, 
    symmectricTSPCost, 
    metropolisCriterion, 
    adjacencyMatrix = adjMat
)

print(symmectricTSPCost(a, adjacencyMatrix = adjMat) )
