# encoding=utf-8

import numpy as ny
import scipy.io as sio

a=[[1,2,3],[4,5,6]]
b=ny.array(a)
c=[0 for i in range(6)]



def load_data(i):
    # 加载数据
    clean_data = sio.loadmat("DecisionTreeData/cleandata_students.mat")
    tdata = clean_data['x']
    ldata = clean_data['y']
    print len(tdata)
    print len(ldata)
    # 处理label
    label_result=[]
    tdata_result=[]
    for ind,label_data in enumerate(ldata):
        if ind % 10 ==i:
            real_label=label_data[0]
            temp_label=[0 for i in range(6)]
            temp_label[real_label-1]=1
            label_result.append(temp_label)

            tdata_result.append(tdata[ind])

    ny_tdata=ny.array(tdata_result)
    ny_label=ny.array(label_result)

    return ny_tdata,ny_label

d,e=load_data(1)

# print b
# print c
print len(d)
print len(e)