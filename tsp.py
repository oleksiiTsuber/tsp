import numpy as np
import math





def transformPointsToDistances(toWorkWith):#a list of point coordinates. The most fucking inefficient thing I've ever written
    def dist(a, b):
        return math.sqrt((a[0] - b[0])** 2 + (a[1] - b[1])**2)

    
    toWorkWith = np.array(toWorkWith)
    n = len(toWorkWith)
    print(n)
    arrayToFill = []

    counter = 1

    for i in range(n-1):
        #print(i)
           
        temp = []
        for j in range(counter, n): #change only this part? 
            temp.append(dist(toWorkWith[i], toWorkWith[j])) 
        temp = [0] * (i+1) + temp

        
        arrayToFill.append(temp)
        counter += 1 #why did I need it? 
    arrayToFill.append(np.zeros(n))   
    print("transformation to an adj matrix is done") 
    return arrayToFill

















def greedySymmetricTSP(adjacencyMatrix, startingNode):#works only with symmetric TSPs
    
    
    
    a = np.triu(adjacencyMatrix)#upper triangle of a matrix
    print(1)
    np.fill_diagonal(a, 0)
    print(2)
    hm = np.nonzero(a)
    print(3)
    newIndices = np.array(list(zip(hm[0], hm[1])))
    print(4)

    a = a.flatten()
    a = a[np.nonzero(a)]
    hm = a.argsort()#I should study this function
    a, newIndices = a[hm], newIndices[hm]
    print(5)
    pizda = 1
    


    
    isUsed = np.zeros(len(adjacencyMatrix))
    path,lengthes = [], []
    counter = 0
    n = len(a)-1

    def getNext(currentNode_, previousEdge_, path_): #so fucking tedious
        
        pizda = 1
        boolMask_ = np.any(np.isin(path_, currentNode_), axis=1)
        pizda = 1
        toConsider_ = np.array(path_)[boolMask_]
        pizda = 1
        hm = [not np.array_equal(previousEdge_, i) for i in toConsider_]#maybe some problems here
        prevEdge_ = toConsider_[hm]
        #prevEdge_ = toConsider_[toConsider_ != previousEdge_]#big problems here. SUKA
        nextNode_ = prevEdge_[prevEdge_ != currentNode_]
        #i'm so tired of this problem
        pizda = 1
        return nextNode_[0], prevEdge_[0]


    pizda = 1
    while counter != n: 
        #print('n, counter: ', n, counter)
        pizda = 1
        thereIsCycle = False#yeah we'll stick to this one


        
        connectivityBeforeChanges = [isUsed[newIndices[counter][0]], isUsed[newIndices[counter][1]]]#just added it. We'll see
        
        
        path.append(newIndices[counter])
        
        if connectivityBeforeChanges == [1, 1]:
            currentNode1 = newIndices[counter][0]
            currentNode2 = newIndices[counter][1]
            #path.append(newIndices[counter]) #if there's a cycle we have to remove it from path
            prevEdge1 = newIndices[counter]
            prevEdge2 = newIndices[counter]
            while True:
                nextNode1, prevEdge1 = getNext(currentNode1, prevEdge1, path)
                nextNode2, prevEdge2 = getNext(currentNode2, prevEdge2, path)
                pizda = 1
                if nextNode1 == currentNode2 or nextNode2 == currentNode1 or nextNode1 == nextNode2:#big problems here
                    thereIsCycle = True 
                    print("CYCLE! CYCLE DETECTED!")
                    break
                if isUsed[nextNode1] <= 1 or isUsed[nextNode2] <= 1:
                    thereIsCycle = False #a bit redundunt
                    break
                currentNode1 = nextNode1
                currentNode2 = nextNode2
            


        
        boolMask = []

        if thereIsCycle:
            path.pop()
        if not thereIsCycle: 
            #path.append(newIndices[counter]) #may be problems
            lengthes.append(a[counter])
            isUsed[newIndices[counter][0]] += 1
            isUsed[newIndices[counter][1]] += 1
            
            if isUsed[newIndices[counter][0]] >= 2:#this thing uses so much memory somehow. Which one though? 
                pass
                boolMask = np.any(np.isin(newIndices, newIndices[counter][0]), axis=1)
                
                boolMask[:counter+1] = False
            
                boolMask = np.invert(boolMask)
                a, newIndices = a[boolMask], newIndices[boolMask]
                #del boolMask #?????
                
            if isUsed[newIndices[counter][1]] >= 2:
                pass
                boolMask = np.any(np.isin(newIndices, newIndices[counter][1]), axis=1)
                boolMask[:counter+1] = False
                boolMask = np.invert(boolMask)
                a, newIndices = a[boolMask], newIndices[boolMask]
                #del boolMask #?????
         
        counter += 1
        n = len(a)-1

    #print("SUKAAAAAAA")
    theLastEdge = np.where(isUsed == 1)[0]
    path.append(theLastEdge)
    lengthes.append(adjacencyMatrix[theLastEdge[0]][theLastEdge[1]])
    

    print("Initial part is done. Now let's sort the path")

    sortedPath = []
    sortedLengthes = []

    def appendToSorted(sortedPath_, sortedLengthes_, prev, path_, lenghtes_):#fucking nested function messing up my scope
        for i in range(len(path_)):
            if prev in path_[i]:
                hm = path_.pop(i)
                if hm[0] != prev:
                    hm[0], hm[1] = hm[1], hm[0]
                sortedPath_.append(hm)
                sortedLengthes_.append(lenghtes_.pop(i))
                break

    appendToSorted(sortedPath, sortedLengthes, startingNode, path, lengthes)
    while len(path) != 0:
        appendToSorted(sortedPath, sortedLengthes, sortedPath[-1][1], path, lengthes)
        
    return {'path': sortedPath, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}




adjacencyMatrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a)

adjacencyMatrix = [[0, 12, 10, 19, 8], [12, 0, 3, 7, 2], [10, 3, 0, 6, 20], [19, 7, 6, 0, 4], [8, 2, 20, 4, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a)


n = 1_000
adjacencyMatrix = np.absolute(np.random.normal(0, 100, n**2)).reshape(n, n)
a = greedySymmetricTSP(adjacencyMatrix, 0)
print(a['all'])


#heeeeeyyyyy
#cnahges are here! 
#sweet new changes. Very good changes