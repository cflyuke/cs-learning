"""counting partitions """
def counting_partitions(n,m):
    if n<0:
        return 0
    elif n==0:
        return 1
    elif m==0:
        return 0
    else:
        return counting_partitions(n-m,m) + counting_partitions(n,m-1) 


# """invert print"""

# def grow(n):
#     if n<10:
#         print(n)
#     else:
#         grow(n//10)
#         print(n)

# def shrink(n):
#     if n>0:
#         print(n)
#         shrink(n//10)

# def inverse_casade(n):
#     grow(n//10)
#     print(n)
#     shrink(n//10)


# """The Luhn Algorithm"""
    
# def split(n):
#     return n // 10, n % 10

# def sum_digits(n):
#     if n < 10:
#         return n
#     else:
#         all_but_last,last=split(n)
#         return sum_digits(all_but_last) + last
    
# def luhn_sum(n):
#     if n<10:
#         return n
#     else:
#         all_but_last,last=split(n)
#         return luhn_sum_double(all_but_last) +last 
    
# def luhn_sum_double(n):
#     all_but_last,last=split(n)
#     return luhn_sum(all_but_last)+sum_digits(2*last)
 