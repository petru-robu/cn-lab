import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

if __name__ == '__main__':
    df = pd.read_csv('./data/iris.csv', header=None)
    y = df.iloc[:, -1].to_numpy()
    A = df.iloc[:, :-1].to_numpy()

    print(A.shape)
    print(A[:5, :])

    for k in range(A.shape[1]):
        A[:, k] = (A[:, k] - np.mean(A[:, k])) / np.std(A[:, k])

    print(A[:5, :])

    comp = 2
    pca = PCA(n_components=comp)
    components = pca.fit_transform(A)

    print(components.shape)
    print(components)

    plt.scatter(components[:, 0], components[:, 1])
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.title('PCA Iris')
    plt.tight_layout()
    plt.savefig('./output/pca.svg')
    plt.show()
    

