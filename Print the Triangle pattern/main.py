# 1. Lower Triangular Pattern

def lower_triangle(n):
    for i in range(1, n + 1):
        print('*' * i)

lower_triangle(5)

# 2. Upper Triangular Pattern

def upper_triangle(n):
    for i in range(n):
        print(' ' * i + '*' * (n - i))

upper_triangle(5)

# 3. Pyramid Pattern

def pyramid(n):
    for i in range(1, n + 1):
        spaces = ' ' * (n - i)
        stars = '*' * (2 * i - 1)
        print(spaces + stars)

pyramid(5)
