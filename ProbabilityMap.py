import numpy as np

class ProbMap:
    def __init__(self, array, r, step):
        self.map = array
        self.step = step
        self.r = r
        self.xx = array.shape[0]
        self.yy = array.shape[1]

    def IsMax(self, i, j):
        label = True
        val = self.map[i,j]
        if val > 0.6:
            for x in range(i-1, i+2):
                for y in range(j-1, j+2):
                    if x < self.xx and y < self.yy and self.map[x,y] > val:
                        label = False
                        break
        else:
            label = False
        return label

    def Up(self, i, j):
        return (i-1, j)
    def Left(self, i, j):
        return (i, j-1)
    def Right(self, i, j):
        return (i, j+1)
    def Down(self, i, j):
        return (i+1, j)

    def MaxMarking(self, maxmap):
        islands = np.zeros(maxmap.shape)
        k = 0
        groups = []
        maxi = maxmap.shape[0]
        maxj = maxmap.shape[1]
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if maxmap[i, j] == 1 and islands[i,j] == 0:
                    k += 1
                    islands[i, j] = k
                    maxmap[i, j] = 2
                    groups.append((i*self.step+self.r, j*self.step+self.r, self.map[i,j]))
                    neighbours = []
                    neighbours.append((i, j))
                    while len(neighbours) > 0:
                        coords = neighbours.pop()
                        x = coords[0]
                        y = coords[1]
                        if self.Left(x, y)[0] < maxi and self.Left(x, y)[1] < maxj and maxmap[self.Left(x, y)] == 1:
                            neighbours.append(self.Left(x, y))
                            maxmap[self.Left(x, y)] = 0
                        if self.Right(x, y)[0] < maxi and self.Right(x, y)[1] < maxj and maxmap[self.Right(x, y)] == 1:
                            neighbours.append(self.Right(x, y))
                            maxmap[self.Right(x, y)] = 0
                        if self.Down(x, y)[0] < maxi and self.Down(x, y)[1] < maxj and maxmap[self.Down(x, y)] == 1:
                            neighbours.append(self.Down(x, y))
                            maxmap[self.Down(x, y)] = 0
                        if self.Up(x, y)[0] < maxi and self.Up(x, y)[1] < maxj and maxmap[self.Up(x, y)] == 1:
                            neighbours.append(self.Up(x, y))
                            maxmap[self.Up(x, y)] = 0
        return k, groups

    def FindMax(self):
        count = 0
        cols = []
        maxmap = np.zeros(self.map.shape)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.IsMax(i, j):
                    count +=1
                    x = i*self.step+self.r
                    y = j*self.step+self.r
                    cols.append((x, y, self.map[i,j]))
                    maxmap[i,j] = 1
        return maxmap

matrix = np.loadtxt("matrix.txt")
prob = ProbMap(matrix, 0, 1)
maxmap = prob.FindMax()
k, gr = prob.MaxMarking(maxmap)
#print(k)
