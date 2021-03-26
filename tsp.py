import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import os


def transformPointsToDistances(toWorkWithFile, name, dirToSave):
    
    toWorkWith = np.delete(np.loadtxt(toWorkWithFile), 0, axis=1)
    n = len(toWorkWith)
    arrayToFill = np.zeros([n,n])

    for i in range(n-1):
        x1 = np.repeat(toWorkWith[i][0], n-(i+1))
        y1 = np.repeat(toWorkWith[i][1], n-(i+1))
        x2 = toWorkWith[i+1:n].T[0]
        y2 = toWorkWith[i+1:n].T[1]
        arrayToFill[i][i+1:n] = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2   )
 
    #np.save('C:\\users\\oleks\\OneDrive\\Рабочий Стол\\tsp\\' + 'test' + '.npy', arrayToFill)
    np.save(dirToSave + name + '.npy', arrayToFill)
    del arrayToFill
    print("Adjacency matrix of " + name +  " is in the file right now")
    return 0






def drawGraph(verticesFile, path, name, dirToSave, ext):
    toWorkWith = np.delete(np.loadtxt(verticesFile), 0, axis=1).T #problems can be here
    x = toWorkWith[0]
    y = toWorkWith[1]
    x = x[path]
    y = y[path]
    plt.plot(x,y, 'k,-', linewidth=0.1)#'k.-'
    plt.savefig(dirToSave + name + ext)
    plt.close()






def greedySymmetricTSP(adjacencyMatrixFile, startingNode):#works only with symmetric TSPs
    
    if type(adjacencyMatrixFile) != str:
        name = os.getcwd()
        np.save(name + 'temp' + '.npy', adjacencyMatrixFile)
        adjacencyMatrixFile = name + 'temp' + '.npy'

    a = np.load(adjacencyMatrixFile)
    matSize = len(a[0])
    #print("a: ", a)
    a = np.triu(a)
    np.fill_diagonal(a, 0)
    newIndices = np.nonzero(a)
    a = a[newIndices]
    newIndices = np.array([newIndices[0], newIndices[1]]).T
    hm = a.argsort()#I should study this function
    newIndices = newIndices[hm]
    #a, newIndices = a[hm], newIndices[hm]
    n = len(a)-1
    del a 
    del hm
    adjListDict = {i : [] for i in range(matSize)}#there are huge problems with init
    isUsed = np.zeros(matSize)
    #path,lengthes = [], []#do we even need path? 
    counter = 0
    print("preprocessing done")

    def getNextNode(currentNode_, prevNode_, adjListDict_):
        for i in adjListDict_[currentNode_]:
            if i != prevNode_:
                return i

    for counter in range(0, n):#while counter != n: 
        #print('n, counter: ', n, counter)
        thereIsCycle = False#yeah we'll stick to this one
        
        connectivityBeforeChanges = [isUsed[newIndices[counter][0]], isUsed[newIndices[counter][1]]]
        
        if isUsed[newIndices[counter][0]] < 2 and isUsed[newIndices[counter][1]] < 2:
            if connectivityBeforeChanges == [1, 1]:

                currentNode1 = newIndices[counter][0]
                currentNode2 = newIndices[counter][1]
                prevNode1 = newIndices[counter][1]
                prevNode2 = newIndices[counter][0]

                while True:
                    pizda = 1
                    nextNode1 = getNextNode(currentNode1, prevNode1, adjListDict) 
                    nextNode2 = getNextNode(currentNode2, prevNode2, adjListDict) 

                    if nextNode1 == currentNode2 or nextNode2 == currentNode1 or nextNode1 == nextNode2:
                        thereIsCycle = True 
                        #print("CYCLE! CYCLE DETECTED!")
                        break
                    if isUsed[nextNode1] <= 1 or isUsed[nextNode2] <= 1:
                        thereIsCycle = False #a bit redundunt
                        break
                    prevNode1 = currentNode1
                    prevNode2 = currentNode2
                    currentNode1 = nextNode1
                    currentNode2 = nextNode2
                
            if thereIsCycle:
                pass
            if not thereIsCycle: 

                isUsed[newIndices[counter][0]] += 1
                isUsed[newIndices[counter][1]] += 1
                pizda = 1

                adjListDict[newIndices[counter][0]].append(newIndices[counter][1])
                adjListDict[newIndices[counter][1]].append(newIndices[counter][0])
                # for sukai in adjListDict:
                #     if len(adjListDict[sukai]) > 2:
                #         print("ALARM ALARM HUGE PROBLEMS AAAAAAA")
                pizda = 1

 

    pizda = 1
    theLastEdge = np.where(isUsed == 1)[0]
    theLastEdge.sort()
    adjListDict[theLastEdge[0]].append(theLastEdge[1])
    adjListDict[theLastEdge[1]].append(theLastEdge[0])
    a = np.load(adjacencyMatrixFile)#or maybe we actually need it
    theLastLength = a[theLastEdge[0]][theLastEdge[1]]
    #del a#????
    #lengthes.append(theLastLength)
    #print("Path: ", path)
    print("The path is formed. Now let's sort it")
    #sortedLengthes = lengthes
    
    prevNode = startingNode
    currentNode = adjListDict[startingNode][0]#let's go in that direction
    sortedNodes = [startingNode, currentNode]
    while currentNode != startingNode: #doesn't work right
        nextNode = getNextNode(currentNode, prevNode, adjListDict)
        prevNode = currentNode
        currentNode = nextNode
        sortedNodes.append(currentNode)

    sortedEdges = [ [sortedNodes[i-1],sortedNodes[i]] for i in range(1, len(sortedNodes)) ]#seems to work allright
    sortedEdgesFromWhichWeCanSample = np.array(sortedEdges.copy())
    sortedEdgesFromWhichWeCanSample.sort(axis=1)
    hm = sortedEdgesFromWhichWeCanSample.T
    sortedLengthes = a[hm[0], hm[1]]#??????
        
    return {'path': sortedEdges, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}




adjacencyMatrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a)

adjacencyMatrix = [[0, 12, 10, 19, 8], [12, 0, 3, 7, 2], [10, 3, 0, 6, 20], [19, 7, 6, 0, 4], [8, 2, 20, 4, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 3)
print(a)


# n = 1_000
# adjacencyMatrix = np.absolute(np.random.normal(0, 100, n**2)).reshape(n, n)
# a = greedySymmetricTSP(adjacencyMatrix, 0)
# print(a['all'])


#heeeeeyyyyy
#cnahges are here! 
#sweet new changes. Very good changes


print(os.getcwd())