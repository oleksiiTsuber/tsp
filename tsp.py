import numpy as np

def greedySymmetricTSP(adjacencyMatrix, startingNode):#works only with symmetric TSPs
    
    
    newIndices = []
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
    


    while counter != n: 
        path.append(newIndices[counter])
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
    
    return {'path': path, 'lengthes': lengthes, 'all': sum(lengthes)}

adjacencyMatrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

a = greedySymmetricTSP(adjacencyMatrix, 0)

print(a)