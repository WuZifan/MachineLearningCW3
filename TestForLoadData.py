# encoding=utf-8

import matplotlib.pyplot as plt

f=open("resultLevel2")
line = f.readline()
count=0
cross_entropy=[]
accuracy=[]
while line:
    if count % 2==0:
        cross_entropy.append(line[15:])
        print line[15:]
    else:
        accuracy.append(line[10:])
        print line[10:]
    count+=1
    line = f.readline()

i=range(len(cross_entropy))
plt.plot(i,cross_entropy,"ro")
# plt.axis([175,190])
plt.show()

f.close
'''
0cross_entropy: 187.766
0accuracy: 0.247054
1cross_entropy: 186.261
1accuracy: 0.251553
2cross_entropy: 185.508
2accuracy: 0.251871
3cross_entropy: 185.136
3accuracy: 0.251911
4cross_entropy: 185.021
4accuracy: 0.251911
5cross_entropy: 184.76
5accuracy: 0.251951
6cross_entropy: 184.585
6accuracy: 0.25199
7cross_entropy: 184.433
7accuracy: 0.25203
8cross_entropy: 184.311
8accuracy: 0.251951
9cross_entropy: 184.233
9accuracy: 0.25199
10cross_entropy: 184.159
10accuracy: 0.25203
11cross_entropy: 184.04
11accuracy: 0.25199
12cross_entropy: 183.941
12accuracy: 0.25199
13cross_entropy: 183.853
13accuracy: 0.25199
14cross_entropy: 183.795
14accuracy: 0.25207
15cross_entropy: 183.743
15accuracy: 0.25207
16cross_entropy: 183.684
16accuracy: 0.25203
17cross_entropy: 183.654
17accuracy: 0.25207
18cross_entropy: 183.588
18accuracy: 0.25207
19cross_entropy: 183.555
19accuracy: 0.25207
20cross_entropy: 183.509
20accuracy: 0.25199
21cross_entropy: 183.424
21accuracy: 0.25203
22cross_entropy: 183.391
22accuracy: 0.25199
23cross_entropy: 183.332
23accuracy: 0.25203
24cross_entropy: 183.298
24accuracy: 0.25199
25cross_entropy: 183.264
25accuracy: 0.25203
26cross_entropy: 183.24
26accuracy: 0.25199
27cross_entropy: 183.201
27accuracy: 0.25203
28cross_entropy: 183.174
28accuracy: 0.25199
29cross_entropy: 183.145
29accuracy: 0.25203
30cross_entropy: 183.128
30accuracy: 0.25199
'''