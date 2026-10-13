import numpy as np

def upper_tri(A, b):
    n = A.shape[0]
    for k in range (0, n-1):
        for i in range(k+1, n):
            gauss_mult = -A[i][k] / A[k][k]
            A[i][k] = 0
            for j in range(k+1, n):
                A[i][j] += A[k][j] * gauss_mult
            
            b[i] += b[k] * gauss_mult
    return A, b

def utris(A, b):
    n = A.shape[0]
    x = np.copy(b)

    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= A[i,j] * x[j]
        x[i] = x[i] / A[i, i]
    return x

    
def main():
    np.set_printoptions(precision=4)

    # random nxn system
    n = 6
    A = np.floor(np.random.rand(n, n) * 100)
    b = np.floor(np.random.rand(n) * 100)
    # print(A, b)

    A, b = upper_tri(A, b)
    # print(A, b)

    x = utris(A, b)
    # print(x)

    print(np.dot(A, x))
    print(b)

    # equal check
    print(np.dot(A, x) - b < 1e9) 

if __name__ == '__main__':
    main()