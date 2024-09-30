import time
import math, numpy, scipy

def timer(func,n,k=None):
    start_time = time.time()
    if k is None:
        func(n)
    else:
        func(n,k)
    display(func, time.time() - start_time)

def display(func, duration):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"function name: {func.__name__} \nduration: {duration}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def factorial_gamma(n):
    temp = math.lgamma(n+1)
    math.exp(temp)

def combination(n,k):
    l1 = math.lgamma(n)
    l2 = math.lgamma(k)
    l3 = math.lgamma(n - k)
    math.exp(l1-l2-l3)

# timer(math.factorial,100)
# timer(factorial_gamma,100)

timer(math.comb,100,50)
timer(combination,100,50)

