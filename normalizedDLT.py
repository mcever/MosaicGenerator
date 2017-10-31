x1 = np.asarray(p1)
m1, s1 = np.mean(x1, 0), math.sqrt(2)
Tr = np.array([[s1, 0, s1*m1[0]], [0, s1, s1*m1[1]], [0, 0, 1]])
Tr = np.linalg.inv(Tr)
x1 = np.dot(Tr, np.concatenate((x1.T, np.ones((1, x1.shape[0])))))
x1 = x1[0:2, :].T

x2 = np.asarray(p2)
m2, s2 = np.mean(x2, 0), math.sqrt(2)
Tr = np.array([[s2, 0, s2*m2[0]], [0, s2, s2*m2[1]], [0, 0, 1]])
Tr = np.linalg.inv(Tr)
x2 = np.dot(Tr, np.concatenate((x2.T, np.ones((1, x2.shape[0])))))
x2 = x2[0:2, :].T

A = []
for i in range(0, len(p1)):
    x = x1[i][0]
    y = x1[i][1]

    u = x2[i][0]
    v = x2[i][1]

    A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

A = np.array(A)
U, S, V = np.linalg.svd(A)
L = V[-1,:] / V[-1,-1]
matrix2 = L.reshape(3, 3)
