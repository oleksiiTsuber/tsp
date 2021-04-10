import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import os


def transformPointsToDistances(toWorkWithFile, name, dirToSave):
    
    toWorkWith = np.delete(np.loadtxt(toWorkWithFile), 0, axis=1)
    toWorkWith = np.unique(toWorkWith, axis=0)
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
    toWorkWith = np.delete(np.loadtxt(verticesFile), 0, axis=1)
    toWorkWith = np.unique(toWorkWith, axis=0).T #problems can be here
    x = toWorkWith[0]
    y = toWorkWith[1]
    x = x[path]
    y = y[path]
    plt.plot(x,y, 'k,-', linewidth=0.1)#'k.-'
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(dirToSave + name + ext)
    plt.close()






def greedySymmetricTSP(adjacencyMatrixFile, startingNode, adjMatTrue):#works only with symmetric TSPs
    
    if type(adjacencyMatrixFile) != str:
        name = os.getcwd()
        np.save(name + 'temp' + '.npy', adjacencyMatrixFile)
        adjacencyMatrixFile = name + 'temp' + '.npy'
    else:
        if adjacencyMatrixFile.split('.')[-1] != 'npy':
            a = np.loadtxt(adjacencyMatrixFile)
            if len(a[0]) == 3:
                a = np.delete(a, 0, axis=1)
            a = np.unique(a, axis=0)
            name = os.getcwd()
            np.save(name + 'temp' + '.npy', a)
            adjacencyMatrixFile = name + 'temp' + '.npy'

    a = np.load(adjacencyMatrixFile)
    
    if adjMatTrue:
        matSize = len(a[0])
        a = np.triu(a)#good code but memory-intensive
        np.fill_diagonal(a, 0)
        newIndices = np.nonzero(a)
        a = a[newIndices]
    else:
        if len(a[0]) == 3:
            a = np.delete(a, 0, axis=1)
        matSize = len(a)
        newIndices = np.zeros((2, sum(range(matSize))), dtype = np.uint16)
        b = np.zeros(sum(range(matSize)))
        counter = 0
        n = matSize

        for i in range(n-1): #should abstract it somehow
            pizda = 1
            x1 = np.repeat(a[i][0], n-(i+1))
            y1 = np.repeat(a[i][1], n-(i+1))
            x2 = a[i+1:n].T[0]
            y2 = a[i+1:n].T[1]
            pizda = 1
            dob = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2   )
            nextCounter = counter+len(dob)
            pizda = 1
            b[counter:nextCounter] = dob #I hope it works
            pizda = 1
            newIndices[0][counter:nextCounter] = np.repeat(i, len(dob))
            pizda = 1
            newIndices[1][counter:nextCounter] = np.arange(i+1, n)
            pizda = 1
            counter = nextCounter

            #del dob, x1, y1, x2, y2#shouldn't work
            #arrayToFill[i][i+1:n] = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2   )
        a = b
        del b
    pizda = 1
    print("got THIS far")
    



    
    hm = a.argsort()#I should study this function
    n = len(a)-1
    del a 
    newIndices = np.array([newIndices[0], newIndices[1]]).T#????????????????????????
    pizda = 1
    newIndices = newIndices[hm]#get as far as here
    #a, newIndices = a[hm], newIndices[hm]
    
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

    for counter in range(0, n):#while counter != n: #should write this thing in C
        #print('n, counter: ', n, counter)
        thereIsCycle = False#yeah we'll stick to this one
        
        connectivityBeforeChanges = [isUsed[newIndices[counter][0]], isUsed[newIndices[counter][1]]]
        pizda = 1
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
    print("The path is formed. Now let's sort it")
    pizda = 1
    #sortedLengthes = lengthes
    
    prevNode = startingNode
    currentNode = adjListDict[startingNode][0]#let's go in that direction
    sortedNodes = [startingNode, currentNode]
    pizda = 1
    while currentNode != startingNode: 
        print(len(sortedNodes))
        pizda = 1
        nextNode = getNextNode(currentNode, prevNode, adjListDict)
        prevNode = currentNode
        currentNode = nextNode
        sortedNodes.append(currentNode)

    sortedEdges = [ [sortedNodes[i-1],sortedNodes[i]] for i in range(1, len(sortedNodes)) ]#seems to work allright
    sortedEdgesFromWhichWeCanSample = np.array(sortedEdges.copy())
    sortedEdgesFromWhichWeCanSample.sort(axis=1) #????????????????????????????????????????????????????????????????????????????????????????????
    hm = sortedEdgesFromWhichWeCanSample.T
    a = np.load(adjacencyMatrixFile)
    if adjMatTrue:
        #a = np.load(adjacencyMatrixFile)#or maybe we actually need it
        sortedLengthes = a[hm[0], hm[1]]#
    else:
        if len(a[0]) == 3:
            a = np.delete(a, 0, axis=1)
        sortedLengthes = []#maybe should've been better np.array
        for i in range(len(sortedEdges)):#maybe inefficient
            x1 = a[sortedEdges[i][0]][0]
            y1 = a[sortedEdges[i][0]][1]
            x2 = a[sortedEdges[i][1]][0]
            y2 = a[sortedEdges[i][1]][1]
            sortedLengthes.append(math.sqrt((x1-x2)**2 + (y1-y2)**2))
            pass
       
    return {'sortedNodes': sortedNodes,  'path': sortedEdges, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}









def symmetricTSPpostProcessing(adjacencyMatrixFile, sortedNodes):
    

    if type(adjacencyMatrixFile) != str:
        name = os.getcwd()
        np.save(name + 'temp' + '.npy', adjacencyMatrixFile)
        adjacencyMatrixFile = name + 'temp' + '.npy'
    else:
        if adjacencyMatrixFile.split('.')[-1] != 'npy':
            a = np.loadtxt(adjacencyMatrixFile)
            if len(a[0]) == 3:
                a = np.delete(a, 0, axis=1)
            a = np.unique(a, axis=0)
            name = os.getcwd()
            np.save(name + 'temp' + '.npy', a)
            adjacencyMatrixFile = name + 'temp' + '.npy'

    adjacencyMatrix = np.load(adjacencyMatrixFile)
    sortedNodes = sortedNodes[:-1] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    positionsDict = {sortedNodes[i] : i for i in range(len(sortedNodes))}#but how will we change it? Do this very thing after every successful insertion? 

    def findNearestNeighbor_(n_, nodesToAvoid_, adjacencyMatrix_):
        arrHorizontal_ = adjacencyMatrix_[n_, : ].copy()
        arrVertical_ = adjacencyMatrix_[ :, n_].copy()
        for i in nodesToAvoid_:
            pizda = 1
            arrHorizontal_[i] = np.inf
            arrVertical_[i] = np.inf #I hope it works
        zerosOnHorisontal_ = n_ + 1
        zerosOnVertical_ = len(adjacencyMatrix_) - n_
        arrHorizontal_[0:zerosOnHorisontal_] = np.repeat(np.inf, zerosOnHorisontal_)
        arrVertical_[n_:len(arrVertical_)] = np.repeat(np.inf, zerosOnVertical_)#I really really hope it will work. Otherwhise I don't know how to test it

        minNodeHorizontal_ = np.argmin(arrHorizontal_)
        minLengthHorizontal_ = np.min(arrHorizontal_)
        minNodeVertical_ = np.argmin(arrVertical_)
        minLengthVertical_ = np.min(arrVertical_)
        minNodes_ = np.array([minNodeHorizontal_, minNodeVertical_])
        minLengthes_ = np.array([minLengthHorizontal_, minLengthVertical_])
        hm_ = np.argsort(minLengthes_)
        minNodes_ = minNodes_[hm_]
        return minNodes_[0]

    def findNearestNeighbors_(n_, adjacencyMatrix_, number_):
        arrHorizontal_ = adjacencyMatrix_[n_, : ].copy()
        arrVertical_ = adjacencyMatrix_[ :, n_].copy()
        arrHorizontal_[n_] = np.inf
        arrVertical_[n_] = np.inf
        zerosOnHorisontal_ = n_ + 1
        zerosOnVertical_ = len(adjacencyMatrix_) - n_
        arrHorizontal_[0:zerosOnHorisontal_] = np.repeat(np.inf, zerosOnHorisontal_)
        arrVertical_[n_:len(arrVertical_)] = np.repeat(np.inf, zerosOnVertical_)
        hmIndices_ = np.concatenate((np.arange(len(adjacencyMatrix)), np.arange(len(adjacencyMatrix))))
        hm2_ = np.concatenate((arrHorizontal_, arrVertical_))
        hmIndices_ = hmIndices_[np.argsort(hm2_)]

        return hmIndices_[0:number_]
    
    
    
    def sample_(nodes_, adjacencyMatrix_):
        hm_ = np.sort(nodes_)
        hm_ = hm_.T 
        pizda = 1
        return adjacencyMatrix_[hm_[0], hm_[1]] #I hope it works
    
    def costFunction_(nodes_, adjacencyMatrix_):
        nodes1_ = nodes_.copy()
        nodes1_.append(nodes1_[0])
        hm_ = np.array([[nodes1_[i-1] , nodes1_[i]] for i in range(1, len(nodes1_))])
        hm_.sort(axis=1)
        hm_ = hm_.T 
        return np.sum(adjacencyMatrix_[hm_[0], hm_[1]])

    number = 20
    nearestNeighbors = {i : findNearestNeighbors_(i, adjacencyMatrix, number) for i in range(len(adjacencyMatrix))}
    pizda = 1
    weAreDone = False
    while not weAreDone:
        weAreDone = True
        for i in range(len(sortedNodes)):
            pizda = 1
            
            neighbors = nearestNeighbors[i].copy()
            nodesToAvoid = [sortedNodes[i], sortedNodes[i-1], sortedNodes[((i+1)%len(sortedNodes))]]
            boolMask = np.repeat(True, len(neighbors))
            for j in range(len(neighbors)):
                if neighbors[j] in nodesToAvoid:
                    boolMask[j] = False
            neighbors = neighbors[boolMask]
            backups = [sortedNodes.copy() for j in range(len(neighbors))]
            costs = np.zeros(len(backups))
            for j in range(len(backups)):
                partOne = backups[j][0:i+1]
                partTwo = backups[j][i+1:len(backups[j])]
                backups[j] = partTwo + partOne
                nn = neighbors[j]
                positionOfnn = backups[j].index(nn)
                partOne = backups[j][0:positionOfnn+1]
                partOne.reverse()
                partTwo = backups[j][positionOfnn+1:len(backups[j])] 
                backups[j] = partTwo + partOne
                costs[j] = costFunction_(backups[j], adjacencyMatrix)
            theSmallest = np.argmin(costs)
            if costs[theSmallest] < costFunction_(sortedNodes, adjacencyMatrix):
                print("HO")
                weAreDone = False
                sortedNodes = backups[theSmallest]

            backup = sortedNodes.copy()#the next part of the code works super-slowly for some reason
            partOne = backup[0:i+1]
            partTwo = backup[i+1:len(backup)]
            pizda = 1
            backup = partTwo + partOne
            nn = findNearestNeighbor_(backup[-1], [backup[-2], backup[0]], adjacencyMatrix)
            pizda = 1
            positionOfnn = backup.index(nn)
            partOne = backup[0:positionOfnn+1]
            partOne.reverse()
            partTwo = backup[positionOfnn+1:len(backup)] 
            backup = partTwo + partOne
            pizda = 1
            if costFunction_(backup, adjacencyMatrix) < costFunction_(sortedNodes, adjacencyMatrix):
                print("HOHO")
                weAreDone = False
                sortedNodes = backup





            # nn = findNearestNeighbor_(sortedNodes[i], [sortedNodes[i-1], sortedNodes[(i+1) % len(sortedNodes)]], adjacencyMatrix)#I hope it works. Maybe the problem's here
            # positionOfnn = positionsDict[nn]
            # backup = sortedNodes.copy()
            # costNow = costFunction_(backup, adjacencyMatrix)
            # variant1 = backup.copy()
            # variant1.insert(positionOfnn-1, variant1.pop(i))
            # pizda = 1
            # costOfVariant1 = costFunction_(variant1, adjacencyMatrix)
            
            # variant2 = backup.copy()
            # variant2.insert(positionOfnn, variant2.pop(i))
            # costOfVariant2 = costFunction_(variant2, adjacencyMatrix)

            # if costOfVariant1 < costNow or costOfVariant2 < costNow:
            #     print("HO")
            #     weAreDone = False
            #     if costOfVariant1 <= costOfVariant2:
            #         sortedNodes = variant1
            #     if costOfVariant2 < costOfVariant1:
            #         sortedNodes = variant2
            #     positionsDict = {sortedNodes[i] : i for i in range(len(sortedNodes))}


            
            # edgesToSubtractInDeletionPlace = np.array([[sortedNodes[i-1], sortedNodes[i]], [sortedNodes[i], sortedNodes[(i+1) % len(sortedNodes)] ] ])
            # edgesToAddInDeletionPlace = np.array([ [sortedNodes[i-1], sortedNodes[(i+1) % len(sortedNodes)]] ])

            # edgesToSubstractFromTheLeft = np.array([ [sortedNodes[positionOfnn-1] , sortedNodes[positionOfnn] ]  ])
            # edgesToAddFromTheLeft = np.array([ [ sortedNodes[positionOfnn-1] , sortedNodes[i] ] ,   [ sortedNodes[i]  ,  sortedNodes[positionOfnn] ] ])

            # edgesToSubstractFromTheRight = np.array([ [sortedNodes[positionOfnn] , sortedNodes[(positionOfnn+1) % len(sortedNodes)] ]  ])
            # edgesToAddFromTheRight = np.array([ [ sortedNodes[positionOfnn] , sortedNodes[i] ] ,   [ sortedNodes[i]  ,  sortedNodes[(positionOfnn+1)% len(sortedNodes)] ] ])

            # lengthesToSubstractInDeletionPlace = sample_(edgesToSubtractInDeletionPlace, adjacencyMatrix)
            # lengthesToAddInDeletionPlace = sample_(edgesToAddInDeletionPlace, adjacencyMatrix)

            # lengthesToSubstractFromTheLeft = sample_(edgesToSubstractFromTheLeft, adjacencyMatrix)
            # lengthesToAddFromTheLeft = sample_(edgesToAddFromTheLeft, adjacencyMatrix)

            # lengthesToSubstractFromTheRight = sample_(edgesToSubstractFromTheRight, adjacencyMatrix)
            # lengthesToAddFromTheRight = sample_(edgesToAddFromTheRight, adjacencyMatrix)

            # variant1 = np.sum(lengthesToAddInDeletionPlace) - np.sum(lengthesToSubstractInDeletionPlace) + np.sum(lengthesToAddFromTheLeft) - np.sum(lengthesToSubstractFromTheLeft)
            # variant2 = np.sum(lengthesToAddInDeletionPlace) - np.sum(lengthesToSubstractInDeletionPlace) + np.sum(lengthesToAddFromTheRight) - np.sum(lengthesToSubstractFromTheRight)

            # if variant1 < 0 or variant2 < 0:
            #     print("HO")
            #     weAreDone = False
            #     if variant1 <= variant2:#from the left
            #         sortedNodes.insert(positionOfnn-1, sortedNodes.pop(i)) #I reaaaaally hope it works 
            #         pass 
            #     if variant2 < variant1:#from the right
            #         sortedNodes.insert(positionOfnn, sortedNodes.pop(i))
            #         pass
            #     positionsDict = {sortedNodes[i] : i for i in range(len(sortedNodes))}
                
    sortedNodes.append(sortedNodes[0])
    sortedEdges = [ [sortedNodes[i-1],sortedNodes[i]] for i in range(1, len(sortedNodes)) ]
    sortedEdgesFromWhichWeCanSample = np.array(sortedEdges.copy())
    sortedEdgesFromWhichWeCanSample.sort(axis=1) #????????????????????????????????????????????????????????????????????????????????????????????
    hm = sortedEdgesFromWhichWeCanSample.T
    sortedLengthes = adjacencyMatrix[hm[0], hm[1]]#
    
    return {'sortedNodes': sortedNodes,  'path': sortedEdges, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}

















adjacencyMatrix = [
    [0, 10, 15, 20]
, [10, 0, 35, 25]
, [15, 35, 0, 30]
, [20, 25, 30, 0]
]

a = greedySymmetricTSP(adjacencyMatrix, 0, True)
print(a)

adjacencyMatrix = [[0, 12, 10, 19, 8], [12, 0, 3, 7, 2], [10, 3, 0, 6, 20], [19, 7, 6, 0, 4], [8, 2, 20, 4, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 3, True)
print(a)


# n = 1_000
# adjacencyMatrix = np.absolute(np.random.normal(0, 100, n**2)).reshape(n, n)
# a = greedySymmetricTSP(adjacencyMatrix, 0, True)
# print(a['all'])




# listOfNodes = [
#     [1, 1], 
#     [1, 5], 
#     [11, 1], 
#     [11, 5]
# ]

# a = greedySymmetricTSP(listOfNodes, 3, False)
# print(a)