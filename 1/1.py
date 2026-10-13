def term(k):
    x_init = 0.1
    
    x = x_init
    for _ in range(k-1):
        print(x)
        if 0 <= x < 0.5:
            x = 2 * x
        elif 1 >= x > 0.5:
            x = 2 * x - 1
    print(f'{x:.4f}')
    return x

if __name__ == '__main__':
    x = term(600)
    print(x)