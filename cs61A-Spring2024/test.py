import numpy as np
import matplotlib.pyplot as plt

# x = np.random.randint(10, 20, size = (3, 3))
# print(x)
# fig = plt.figure(figsize = (10, 5) )
# plt.plot(x.mean(axis = 1), color = "tab:blue", label = "row")
# plt.plot(x.mean(axis = 0),color = "tab:red", label = "col")
# plt.plot(x, alpha=0.1, color = "tab:blue")
# plt.legend()
# plt.show()



#generator function which not return but yield, each time you call the generator, it will
#estabish a generator which can help yiield over the generator function

#tree
# def tree(label, branches = []):
#     for branch in branches:
#         assert is_tree(branch)
#     return [label] + branches

# def label(tree):
#     return tree[0]

# def branches(tree):
#     return tree[1:]

# def is_leaf(tree):
#     return not branches(tree)

# def is_tree(tree):
#     if type(tree) != list or len(tree) == 0:
#         return False
#     for branch in branches(tree):
#         if not is_tree(branch):
#             return False
#     return True

# def count_leaves(tree):
#     if is_leaf(tree):
#         return 1
#     else :
#         leaves = [count_leaves(branch) for branch in branches(tree)]
#         return sum(leaves)

# def count_paths(t, total):
#     if label(t) == total:
#         found = 1
#     else :
#         found = 0
#     return found + sum ([count_paths(x, total - label(t)) for x in branches(t)])


    # sum = 0
    # def counter(t, sum_tem):
    #     sum_tem = sum_tem + label(t)
    #     if sum_tem == total:
    #         sum += 1
    #     elif sum_tem < total:
    #         for branch in branches(t):
    #             counter(branch, sum_tem)
    # counter(t, 0)
    # return sum
# data abstraction
# 一些感触：
## 1. python的数据抽象和java感觉非常像，例如他们的constructor函数其实是一致的，
##    然后python的selector函数其实就很像java中的类函数，所需要的就是对象的数据，然后可以去return对象的信息
##    java中的类方法其实也是在去得到目标的一些特性，和selector函数要实现的功能非常像，但类方法还可以去改变对象的信息
## 2. 这样去写代码的好处就是能够建立起抽象屏障，每个模块都不需要管前者是怎么实现的，
##    只需知道他的提供给你的是按照正确的逻辑给出的，进行操作。从而整个逻辑非常清楚
##    所以在java程序中吗我们应当尽量避免去直接去创造实例字段，而去用方法来表示出来对应的数据，除非该数据在多个函数中使用
## 3. 相像的原因本质上还是因为他们都在对各种数据进行抽象，从而把他们隔开，相互独立。

# constructor and selector for rational number
# def rational(n, d):
#     def select(name):·
#         if name == 'n':
#             return n
#         elif name == 'd':
#             return d
#     return select

# def numer(x):
#     return x('n')

# def demor(x):
#     return x('d')

# def add_rational(x, y):
#     nx, dy = numer(x)*demor(y) + numer(y)*demor(x), demor(x)*demor(y)
#     return rational(x,y)

# 常见的几种容器：list，string，numerals（map）
# you can't repeat a key ; key can't be a list and dictionary
# sum,max,all用于处理list

# """count the number of times that value appears in sequence s"""
# def count(s,value):
#     total = 0
#     for element in s:
#         if element  == value:
#             total+=1
#     return total

# def sum_below(n):
#     total = 0
#     for i in range(n):
#         total += i
#         return total

# """counting partitions """
# def counting_partitions(n,m):
#     if n<0:
#         return 0
#     elif n==0:
#         return 1
#     elif m==0:
#         return 0
#     else:
#         return counting_partitions(n-m,m) + counting_partitions(n,m-1) 


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
 