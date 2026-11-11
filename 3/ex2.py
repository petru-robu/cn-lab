import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

if __name__ == '__main__':
    A = plt.imread('data/Baboon.bmp')
    print(A.shape)
    plt.imshow(A, cmap='gray')
    plt.show()

    # valorile singulare
    U, S, Vt = np.linalg.svd(A)
    print(f'Valorile singulare: {S}\nShape: {S.shape}') 

    for k in [1, 3, 5, 10, 25, 50, 100]:
        aprox = U[:, 0:k] @ np.diag(S[:k]) @ Vt[0:k, :]

        str = f'rank_{k}_aprox'
        print(f'{str}: {aprox}')

        plt.title(str)
        plt.imshow(aprox, cmap='gray')
        plt.savefig('output/' + str + '.svg')
        plt.show()
        

    