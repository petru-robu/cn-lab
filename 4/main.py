import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=2)


if __name__ == '__main__':
    df = pd.read_csv('./data/iris.csv', header=None)
    y = df.iloc[:, -1].to_numpy()
    A = df.iloc[:, :-1].to_numpy()
    
    n_train, n_test = 100, 50
    n = n_train + n_test
    A_train, A_test, y_train, y_test = train_test_split(A, y, train_size=n_train, stratify=y)
    
    print(A_train.shape, A_test.shape)
    print(y_test.shape)

    print('With linear kernel: ')

    clf = SVC(kernel='linear')
    clf.fit(A_train, y_train)
    y_pred = clf.predict(A_test)

    acc = np.mean(y_pred == y_test)
    print(f'Accuracy: {acc:.4f}')

    cls = np.unique(y)
    conf_matrix = np.zeros((len(cls), len(cls)))
    k = len(cls)
    idx = {c: i for i, c in enumerate(cls)}
    for t, p in zip(y_test, y_pred):
        conf_matrix[idx[t], idx[p]] += 1

    print('Classes: ', cls)
    print('Conf matrix: \n', conf_matrix)

    print('With RBF kernel: ')

    clf = SVC(kernel='rbf')
    clf.fit(A_train, y_train)
    y_pred = clf.predict(A_test)

    acc = np.mean(y_pred == y_test)
    print(f'Accuracy: {acc:.4f}')

    cls = np.unique(y)
    conf_matrix = np.zeros((len(cls), len(cls)))
    k = len(cls)
    idx = {c: i for i, c in enumerate(cls)}
    for t, p in zip(y_test, y_pred):
        conf_matrix[idx[t], idx[p]] += 1

    print('Classes: ', cls)
    print('Conf matrix: \n', conf_matrix)