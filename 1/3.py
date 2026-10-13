import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def main():
    reg_data = np.genfromtxt('data/regresie.csv', delimiter=',')
    print(reg_data.shape)
    # print(reg_data)

    z = reg_data[:,:1]
    # print(z)

    A = np.column_stack([z, np.ones((z.shape[0], 1))])
    print(A)

    b = reg_data[:,1:2]
    print(b)

    x, _, _, _ = linalg.lstsq(A, b)
    print(x)

    plt.scatter(z, b, color='orange')

    x_line = np.linspace(-2, 1, 40)
    y_line = x[0] * x_line + x[1]

    plt.plot(x_line, y_line)
    plt.savefig('output/reg.svg')
    plt.show()

if __name__ == '__main__':
    main()