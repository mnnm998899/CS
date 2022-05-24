# Data analysis % wrangling
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
train = pd.read_csv(r"train.csv")
test = pd.read_csv(r"test.csv")

# Split training dataset into binary form (2 classes each dataset)
train_12= train.loc[train['class'].isin(["class-1","class-2"])] #Iloc=int location,,, get class-1 and -2
train_12["class"] = train_12["class"].map({"class-1":1, "class-2":-1})
train_12 = train_12.values
np.random.shuffle(train_12)
#---------------------------------------------#
train_23= train.loc[train['class'].isin(["class-2","class-3"])] #Iloc=int location,,, get class-1 and -2
train_23["class"] = train_23["class"].map({"class-2":1, "class-3":-1})
train_23 = train_23.values
np.random.shuffle(train_23)
#---------------------------------------------#
train_13= train.loc[train['class'].isin(["class-1","class-3"])] #Iloc=int location,,, get class-1 and -2
train_13["class"] = train_13["class"].map({"class-1":1, "class-3":-1})
train_13 = train_13.values
np.random.shuffle(train_13)
#-------------------Testing data-------------------------#
test_12= test.loc[test['class'].isin(["class-1","class-2"])]
test_12["class"] = test_12["class"].map({"class-1":1, "class-2":-1})
test_12 = test_12.values
test_12y=test_12[:,4]
test_12=test_12[:,0:4]
np.array([test_12])
print(f"____--------------____----{test_12.shape}")
#---------------------------------------------#
test_23= test.loc[test['class'].isin(["class-2","class-3"])]
test_23["class"] = test_23["class"].map({"class-2":1, "class-3":-1})
test_23 = test_23.values
test_23y=test_23[:,4]
test_23=test_23[:,0:4]
#---------------------------------------------#
test_13= test.loc[test['class'].isin(["class-1","class-3"])]
test_13["class"] = test_13["class"].map({"class-1":1, "class-3":-1})
test_13 = test_13.values
test_13y=test_13[:,4]
test_13=test_13[:,0:4]


#error for training
err_12=np.zeros(30)
err_13=np.zeros(30)
err_23=np.zeros(30)
#error for testing
errr_12=np.zeros(30)
errr_13=np.zeros(30)
errr_23=np.zeros(30)
w_12=np.array([0,0,0,0])
b_12=0


w_13=np.array([0,0,0,0])
b_13=0


w_23=np.array([0,0,0,0])
b_23=0

def test(set,f):
    if set =="class_12":
        w=w_12
        b=b_12
    elif set=="class_13":
        w=w_13
        b=b_13
    elif set=="class_23":
        w=w_23
        b=b_23
    else:
        print("enter a prpoer value")

        
    
    a=np.dot(w,f)+b
    if a>0:
        print("class 1")
    else:
        print("class 2")
    return a

it_number=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]# itteration number

for s in range(len(it_number)):
    print(f" itteration no.{s}")
#-------Training_12----------------------------#
    x_12=train_12[:,0:4]
    y_12=train_12[:,4]
    i_12=0
    e_12=0
    
    for k in range(0,len(x_12)):
        a_12=np.dot(x_12[i_12,:],w_12.T)
        c_12=a_12*y_12[i_12]
    # print("itteraion number %d" %(i))
    # print("Actvation={act}".format(act=a))
    # print("The correct y={Cor}".format(Cor=y_12[i]))
    # print("a*y={Thr}".format(Thr=c))

        if c_12<=0:
        # print("Missclassfied")
        # print(y_12[i])
        # print(x_12[i,:])
            w_12=w_12+y_12[i_12]*x_12[i_12,:]
            b_12=b_12+y_12[i_12]
            e_12+=1
        # print("e={e}".format(e=e))
    # else:
    #     # print("True")
       
        i_12+=1
    # print("W={weight}" .format(weight=w_12))
    # print("b= %d" %(b_12))


    ac_12=(1-(e_12/i_12))*100
    print("The accuracy={rf}".format(rf=ac_12))
    # print(e_12)
    # print(i_12)
    err_12[s]=e_12
#-------Training_13----------------------------#
    x_13=train_13[:,0:4]
    y_13=train_13[:,4]
    i_13=0
    e_13=0

    for k in range(0,len(x_13)):
        a_13=np.dot(x_13[i_13,:],w_13.T)
        c_13=a_13*y_13[i_13]
    # print("itteraion number %d" %(i_13))
    # print("Actvation={act_13}".format(act_13=a_13))
    # print("The correct y={Cor_13}".format(Cor_13=y_13[i_13]))
    # print("a*y={Thr_13}".format(Thr_13=c_13))

        if c_13<=0:
        # print("Missclassfied")
        # print(y_13[i_13])
        # print(x_13[i_13,:])
            w_13=w_13+y_13[i_13]*x_13[i_13,:]
            b_13=b_13+y_13[i_13]
            e_13+=1
        # print("e={e}".format(e=e))
    # else:
    #     print("True")
       
        i_13+=1
    # print("W={weight_13}" .format(weight_13=w_13))
    # print("b= %d" %(b_13))


    ac_13=(1-(e_13/i_13))*100
    print("The accuracy={rf_13}".format(rf_13=ac_13))
    # print(e_13)
    # print(i_13)
    err_13[s]=e_13
#-------Training_23----------------------------#
    x_23=train_23[:,0:4]
    y_23=train_23[:,4]
    i_23=0
    e_23=0


    
    for k in range(0,len(x_23)):
        a_23=np.dot(x_23[i_23,:],w_23.T)
        c_23=a_23*y_23[i_23]
    # print("itteraion number %d" %(i_13))
    # print("Actvation={act_13}".format(act_13=a_13))
    # print("The correct y={Cor_13}".format(Cor_13=y_13[i_13]))
    # print("a*y={Thr_13}".format(Thr_13=c_13))

        if c_23<=0:
            w_23=w_23+y_23[i_23]*x_23[i_23,:]
            b_23=b_23+y_23[i_23]
            e_23+=1
    
    # else:
    # #     print("True")
       
        i_23+=1
    # print("W={weight_13}" .format(weight_13=w_13))
    # print("b= %d" %(b_13))


    ac_23=(1-(e_23/i_23))*100
    print("The accuracy={rf_23}".format(rf_23=ac_23))
    err_23[s]=e_23
    # print(w_12)
    # print(b_12)
    # print(err_12)
    # print(err_13)
    # print(err_23)
    ee_12=0
    ee_13=0
    ee_23=0
    #>>>>>>>>>>>>>Testing Class_12>>>>>>>>>>>>>>>>>>>>>>>>>
    for n in range(len(test_12y)):
        activation=test("class_12",test_12[n,:])*test_12y[n]
        if activation<=0:
            ee_12+=1
    errr_12[s]=ee_12
    print(f"the no.{errr_12}")
#>>>>>>>>>>>>>Testing Class_13>>>>>>>>>>>>>>>>>>>>>>>>>
    for n in range(len(test_13y)):
        activation=test("class_13",test_13[n,:])*test_13y[n]
        if activation<=0:
            ee_13+=1
    errr_13[s]=ee_13
    print(f"the no. for Class13{errr_13}")

#>>>>>>>>>>>>>Testing Class_13>>>>>>>>>>>>>>>>>>>>>>>>>
    for n in range(len(test_23y)):
        activation=test("class_23",test_23[n,:])*test_23y[n]
        if activation<=0:
            ee_23+=1
    errr_23[s]=ee_23
    print(f"the no. for Class_23{errr_23}")

print(test("class_12",test_12[0,:]))

# print(test("class_12",4.6,3.2,1.4,0.2))
# test("class_12",5.7,3.0,4.2,1.2)

# plt.plot(it_number,err_12)
# plt.plot(it_number,err_13)
# plt.plot(it_number,err_23)

plt.plot(it_number,errr_12)
plt.plot(it_number,errr_13)
plt.plot(it_number,errr_23)

plt.legend(["testing error for class_12","testing error for class_13","testing error for class_23"])
#,"testing error for class_12","testing error for class_13","testing error for class_23"
plt.show()
print(ee_12)
print(ee_13)
print(ee_23)
