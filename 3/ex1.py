import numpy as np
np.set_printoptions(precision=2)


if __name__ == '__main__':
    # matrice random
    m, r = 10, 4
    A = np.random.random((m, r))
    rA = np.linalg.matrix_rank(A)
    print(f'Matricea: {A}\nShape: {A.shape} Rank: {rA}')

    # adaugam coloane liniar dependente
    n = 8
    dcol = n - r
    c = 3
    for nc_idx in range(dcol):
        rnd_indices = np.random.randint(0, r, 3)
        rnd_cols = A[:, rnd_indices]
        rnd_coeffs = np.random.randn(rnd_cols.shape[1])
        new_col = rnd_cols @ rnd_coeffs
        A = np.column_stack((A, new_col))
        print(f'Added new col({nc_idx}) based on cols at {rnd_indices}, the new col is: {new_col}')

    print(f'Noua matrice: {A}\nShape: {A.shape} Rank: {np.linalg.matrix_rank(A)}')

    # adaugam zgomot gaussian si calc. rangul
    med, disp =  0, 0.2
    noise = np.random.normal(med, disp, A.shape)
    A = A + noise
    print(f'Noua matrice dupa zgomot: {A}\nShape: {A.shape} Rank: {np.linalg.matrix_rank(A)}')

    # valorile singulare
    U, S, V = np.linalg.svd(A)
    print(f'Valorile singulare: {S}') 

    # recuperam rangul real
    # valorile singulare care sunt neglijabile sunt cele care sunt 
    # neglijabile in raport cu zgomotul, pentru zgomot gauassian putem 
    # compara cu std*(sqrt(m) + sqrt(n))

    thr = disp * (np.sqrt(m) + np.sqrt(n))
    real_rank = np.sum(S > thr)
    print(f'Rangul recuperat: {real_rank}') # pe langa 4 (rangul corect), obtin cateodata si 3 si 2