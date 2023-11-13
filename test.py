#
# x = "Wednesday"
# y = x [1:5][100:500] + x [2::4]
# print(y)
#
# z = "university" #ytisrevinu
# print (z [::-1][1:4])


# x = {(10,), (20,)}
# print(x)
# # y = {[10], [20]}
# # print(y)
# z = {10, 20}
# print(z)
#
# x = [[10], 20]
# y = x.copy()
# y[0][0] = 100
# print(x, y)
#
# m = {10: 20, 30: 40}
# for x, y in m.items():
#     print(x, y, end=" ")

# def f(n):
#     if n == 1:
#         return 1
#     else:
#         return 2 ** n * f(n - 1)
#
# print(f(3))
#
# # f(3) = 2^3 *f(2) 8 * 4
# # f(2) = 2^2 * f(1) = 4*1
# # f(1) = 1


x = [10 , 20, 30]
y = x
print(id(x) == id(y))
