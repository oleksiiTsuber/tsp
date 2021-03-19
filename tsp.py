import numpy as np

def greedySymmetricTSP(adjacencyMatrix, startingNode):#works only with symmetric TSPs
    
    
    newIndices = []#what is this variable, for god's sake? Index of every enge in the adjacencyMatrix it seems
    for i in range(len(adjacencyMatrix)):
        for j in range(len(adjacencyMatrix[i])):
            newIndices += [[i,j]]
    newIndices = np.array(newIndices)      
    
    a = np.triu(adjacencyMatrix)#upper triangle of a matrix
    np.fill_diagonal(a, 0)
    a = a.flatten()
    
    hm = a.argsort()#I should study this function
    a, newIndices = a[hm], newIndices[hm]
    
    hm = np.nonzero(a)
    a, newIndices = a[hm], newIndices[hm]
    
    isUsed = np.zeros(len(adjacencyMatrix))
    path,lengthes = [], []
    counter = 0
    n = len(a)-1
    
    def getNext(currentNode_, previousEdge_, path_): #so fucking tedious
        #find an edge in the pass which contains the current node
        #return the other node in that edge
        
        boolMask_ = np.any(np.isin(path_, currentNode_), axis=1)
        
        toConsider_ = np.array(path_)[boolMask_]
        prevEdge_ = toConsider_[toConsider_ != previousEdge_]
        nextNode_ = prevEdge_[prevEdge_ != currentNode_]
        #i'm so tired of this problem
        
        return nextNode_, prevEdge_


    
    while counter != n: 
        

        thereIsCycle = False#yeah we'll stick to this one


        #pathUnderConsideration = newIndices[counter]#just to watch yet
        connectivityBeforeChanges = [isUsed[newIndices[counter][0]], isUsed[newIndices[counter][1]]]#just added it. We'll see
        # isUsed[newIndices[counter][0]] += 1
        # isUsed[newIndices[counter][1]] += 1
        
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
                if nextNode1 == currentNode2 or nextNode2 == currentNode1 or nextNode1 == nextNode2:
                    thereIsCycle = True 
                    break
                if isUsed[nextNode1] <= 1 or isUsed[nextNode2] <= 1:
                    thereIsCycle = False #a bit redundunt
                    break
                currentNode1 = nextNode1
                currentNode2 = nextNode2
            


        


        if thereIsCycle:
            # isUsed[newIndices[counter][0]] -= 1
            # isUsed[newIndices[counter][1]] -= 1
            path.pop()
        if not thereIsCycle: 
            #path.append(newIndices[counter]) #may be problems
            lengthes.append(a[counter])
            isUsed[newIndices[counter][0]] += 1
            isUsed[newIndices[counter][1]] += 1
            
            if isUsed[newIndices[counter][0]] >= 2:
                boolMask = np.any(np.isin(newIndices, newIndices[counter][0]), axis=1)
                
                boolMask[:counter+1] = False
            
                boolMask = np.invert(boolMask)
                a, newIndices = a[boolMask], newIndices[boolMask]
                
            if isUsed[newIndices[counter][1]] >= 2:
                boolMask = np.any(np.isin(newIndices, newIndices[counter][1]), axis=1)
                boolMask[:counter+1] = False
                boolMask = np.invert(boolMask)
                a, newIndices = a[boolMask], newIndices[boolMask]
         
        counter += 1
        n = len(a)-1

    
    theLastEdge = np.where(isUsed == 1)[0]
    path.append(theLastEdge)
    lengthes.append(adjacencyMatrix[theLastEdge[0]][theLastEdge[1]])
    #now we should sort by the starting node i guess. Though it shouldn't matter

    path1 = path.copy()
    lengthes1 = lengthes.copy()

    sortedPath = []
    sortedLengthes = []

    def appendToSorted(sortedPath, sortedLengthes, prev, path_, lenghtes_):#fucking nested function messing up my scope
        for i in range(len(path_)):
            if prev in path_[i]:
                hm = path_.pop(i)
                if hm[0] != prev:
                    hm[0], hm[1] = hm[1], hm[0]
                sortedPath.append(hm)
                sortedLengthes.append(lenghtes_.pop(i))
                break

    appendToSorted(sortedPath, sortedLengthes, startingNode, path1, lengthes1)
    while len(path1) != 0:
        appendToSorted(sortedPath, sortedLengthes, sortedPath[-1][1], path1, lengthes1)
        
    return {'path': sortedPath, 'lengthes': sortedLengthes, 'all': sum(sortedLengthes)}




adjacencyMatrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

adjacencyMatrix = [[0, 12, 10, 19, 8], [12, 0, 3, 7, 2], [10, 3, 0, 6, 20], [19, 7, 6, 0, 4], [8, 2, 20, 4, 0]]



a = greedySymmetricTSP(adjacencyMatrix, 0)

print(a)


#heeeeeyyyyy
#cnahges are here! 
#sweet new changes. Very good changes