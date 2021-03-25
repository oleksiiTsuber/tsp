import numpy as np
import math
import pandas as pd
import os


def transformPointsToDistances(toWorkWith, name, dir):#a list of point coordinates. The most fucking inefficient thing I've ever written
    def dist(a, b):
        return math.sqrt((a[0] - b[0])** 2 + (a[1] - b[1])**2)
    
    toWorkWith = np.array(toWorkWith)
    n = len(toWorkWith)
    arrayToFill = np.zeros([n,n])

    for i in range(n-1):
        #print(i) 
        x1 = np.repeat(toWorkWith[i][0], n-(i+1))
        y1 = np.repeat(toWorkWith[i][1], n-(i+1))
        x2 = toWorkWith[i+1:n].T[0]
        y2 = toWorkWith[i+1:n].T[1]
        pizda = 1
        suka = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2   )
        pizda = 1
        arrayToFill[i][i+1:n] = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2   )
 
    #np.save('C:\\users\\oleks\\OneDrive\\Рабочий Стол\\tsp\\' + 'test' + '.npy', arrayToFill)
    np.save(dir + name + '.npy', arrayToFill)
    return "Adjacency matrix is in the file right now"

















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
    a, newIndices = a[hm], newIndices[hm]
    
    pizda = 1



    adjListDict = {i : [] for i in range(matSize)}#there are huge problems with init


    isUsed = np.zeros(matSize)
    path,lengthes = [], []
    counter = 0
    n = len(a)-1 
    print("preprocessing done")



    # def getNext(currentNode_, previousEdge_, path_): #so fucking tedious. And problematic. I should consider using an adjList for this
        
    #     pizda = 1
    #     boolMask_ = np.any(np.isin(path_, currentNode_), axis=1)
    #     pizda = 1
    #     toConsider_ = np.array(path_)[boolMask_]
    #     pizda = 1
    #     hm = [not np.array_equal(previousEdge_, i) for i in toConsider_]
    #     prevEdge_ = toConsider_[hm]
    #     #prevEdge_ = toConsider_[toConsider_ != previousEdge_]#big problems here. SUKA
    #     nextNode_ = prevEdge_[prevEdge_ != currentNode_]
    #     #i'm so tired of this problem
    #     pizda = 1
    #     return nextNode_[0], prevEdge_[0]

    def getNextNode(currentNode_, prevNode_, adjListDict_):
        for i in adjListDict_[currentNode_]:
            if i != prevNode_:
                return i



    pizda = 1
    while counter != n: 
        #print('n, counter: ', n, counter)
        pizda = 1
        thereIsCycle = False#yeah we'll stick to this one


        
        connectivityBeforeChanges = [isUsed[newIndices[counter][0]], isUsed[newIndices[counter][1]]]#just added it. We'll see
        
        if isUsed[newIndices[counter][0]] < 2 and isUsed[newIndices[counter][1]] < 2:

            path.append(newIndices[counter])#under ??? right now


            
            if connectivityBeforeChanges == [1, 1]:

                currentNode1 = newIndices[counter][0]
                currentNode2 = newIndices[counter][1]
                prevNode1 = newIndices[counter][1]
                prevNode2 = newIndices[counter][0]

                while True:
                    pizda = 1
                    nextNode1 = getNextNode(currentNode1, prevNode1, adjListDict) 
                    nextNode2 = getNextNode(currentNode2, prevNode2, adjListDict) 

                    if nextNode1 == currentNode2 or nextNode2 == currentNode1 or nextNode1 == nextNode2:#big problems here
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



                # currentNode1 = newIndices[counter][0]
                # currentNode2 = newIndices[counter][1]
                # #path.append(newIndices[counter]) #if there's a cycle we have to remove it from path
                # prevEdge1 = newIndices[counter]
                # prevEdge2 = newIndices[counter]
                # while True:
                #     nextNode1, prevEdge1 = getNext(currentNode1, prevEdge1, path)
                #     nextNode2, prevEdge2 = getNext(currentNode2, prevEdge2, path)
                #     pizda = 1
                #     if nextNode1 == currentNode2 or nextNode2 == currentNode1 or nextNode1 == nextNode2:#big problems here
                #         thereIsCycle = True 
                #         #print("CYCLE! CYCLE DETECTED!")
                #         break
                #     if isUsed[nextNode1] <= 1 or isUsed[nextNode2] <= 1:
                #         thereIsCycle = False #a bit redundunt
                #         break
                #     currentNode1 = nextNode1
                #     currentNode2 = nextNode2
                
            if thereIsCycle:
                path.pop()
            if not thereIsCycle: 
                #path.append(newIndices[counter]) #may be problems
                lengthes.append(a[counter])
                isUsed[newIndices[counter][0]] += 1
                isUsed[newIndices[counter][1]] += 1
                pizda = 1

                adjListDict[newIndices[counter][0]].append(newIndices[counter][1])
                adjListDict[newIndices[counter][1]].append(newIndices[counter][0])
                # for sukai in adjListDict:
                #     if len(adjListDict[sukai]) > 2:
                #         print("ALARM ALARM HUGE PROBLEMS AAAAAAA")
                pizda = 1
                pass
                
         
        counter += 1
        n = len(a)-1

    pizda = 1
    theLastEdge = np.where(isUsed == 1)[0]
    theLastEdge.sort()
    path.append(theLastEdge)
    adjListDict[theLastEdge[0]].append(theLastEdge[1])
    adjListDict[theLastEdge[1]].append(theLastEdge[0])
    a = np.load(adjacencyMatrixFile)#or maybe we actually need it
    theLastLength = a[theLastEdge[0]][theLastEdge[1]]
    #del a#????
    lengthes.append(theLastLength)
    pizda = 1
    #print("Path: ", path)
    print("The path is formed. Now let's sort it")
    #sortedLengthes = lengthes
    


    prevNode = startingNode
    currentNode = adjListDict[startingNode][0]#let's go in that direction
    sortedNodes = [startingNode, currentNode]
    pizda = 1
    while currentNode != startingNode: #doesn't work right
        pizda = 1
        nextNode = getNextNode(currentNode, prevNode, adjListDict)
        prevNode = currentNode
        currentNode = nextNode
        sortedNodes.append(currentNode)
        pass


    pizda = 1


    sortedEdges = [ [sortedNodes[i-1],sortedNodes[i]] for i in range(1, len(sortedNodes)) ]#seems to work allright
    sortedEdgesFromWhichWeCanSample = np.array(sortedEdges.copy())
    sortedEdgesFromWhichWeCanSample.sort(axis=1)
    pizda  = 1
    hm = sortedEdgesFromWhichWeCanSample.T
    pizda = 1
    sortedLengthes = a[hm[0], hm[1]]#??????
    pizda = 1
    #print("SortedNodes: ", sortedNodes)
    #print("SortedEdges: ", sortedEdges)
    # sortedPath = []
    # sortedLengthes = []

    # def appendToSorted(sortedPath_, sortedLengthes_, prev, path_, lenghtes_):#fucking nested function messing up my scope
    #     for i in range(len(path_)):
    #         if prev in path_[i]:
    #             hm = path_.pop(i)
    #             if hm[0] != prev:
    #                 hm[0], hm[1] = hm[1], hm[0]
    #             sortedPath_.append(hm)
    #             sortedLengthes_.append(lenghtes_.pop(i))
    #             break

    # appendToSorted(sortedPath, sortedLengthes, startingNode, path, lengthes)
    # while len(path) != 0:
    #     appendToSorted(sortedPath, sortedLengthes, sortedPath[-1][1], path, lengthes)
        
    return {'path': sortedEdges, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}




adjacencyMatrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a)

adjacencyMatrix = [[0, 12, 10, 19, 8], [12, 0, 3, 7, 2], [10, 3, 0, 6, 20], [19, 7, 6, 0, 4], [8, 2, 20, 4, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 3)
print(a)


n = 1_000
adjacencyMatrix = np.absolute(np.random.normal(0, 100, n**2)).reshape(n, n)
a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a['all'])


#heeeeeyyyyy
#cnahges are here! 
#sweet new changes. Very good changes


print(os.getcwd())