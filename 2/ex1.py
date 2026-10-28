import numpy as np

def power_eigen_values(A, tol, max_iter):
    n, m = A.shape[0], A.shape[1]
    if n != m:
        print("Can't compute eigenvals!")
        return None

    y = np.random.rand(n, 1)
    # print(f'Initial rnd values {y.shape}: {y}')

    err, iter = 1, 0
    while err > tol:
        if iter > max_iter:
            break

        y = y / np.linalg.norm(y)
        z = np.dot(A, y)
        z = z / np.linalg.norm(z)

        err = np.abs(1 - np.abs(np.dot(z.T, y)))

        y = z.copy()
        iter += 1

    return y.T


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    A = np.array([[1, 20, 3, 4],
                 [20, 1, 2, 3],
                 [3, 2, 3, 4],
                 [4, 4, 4, 2]])

    dom_eig_vec = power_eigen_values(A, 1e-9, 1000)
    dom_eig_val = (dom_eig_vec @ A @ dom_eig_vec.T).item()

    print(f'Dominant eigenvector using power method: {dom_eig_vec}')
    print(f'Dominant eigenvalue using power method: {dom_eig_val}')

    eig_vals_np, eig_vec_np = np.linalg.eig(A)
    dom_eig_val_np = np.max(np.abs(eig_vals_np))

    print(f'Dominant eigenvalue from np: {dom_eig_val_np}')

    diff = dom_eig_val_np - dom_eig_val
    print(f'Diff: {diff}')

    tol = 1e-5
    if diff < tol:
        print(f'Matching eigenvals! (tol: {tol})')


