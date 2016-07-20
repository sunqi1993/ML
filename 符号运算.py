k=[1,1,1,0]
s=[1,1,0,0]
s=[int(x==y) for x,y in zip(k,s)]
print(sum(s))