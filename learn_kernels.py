import pdb
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
# from utilities3 import LpLoss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
from functools import reduce
from functools import partial
import operator
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter
import deepxde as dde
import time


# Build out grid
# Parameters
epochs = 800
ntrain = 1800
ntest = 200
batch_size = 32
gamma = 0.5
learning_rate = 0.001
step_size= 50
modes=12
width=32



X = 1
dx = 0.005
nx = int(round(X/dx))
spatial = np.linspace(0, X, nx, dtype=np.float32)

grids = []
grids.append(spatial)
grids.append(spatial)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid1 = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
print(grid1.shape)
grid = np.concatenate([grid], axis=1)
grid = torch.from_numpy(grid).cuda()

y_K= np.loadtxt("y_L2_2000_18.dat", dtype=np.float32)
# y_K= np.loadtxt("y_K2_2000_18.dat", dtype=np.float32)
# y_K= np.loadtxt("y_m12_2000_18.dat", dtype=np.float32)
# y_K= np.loadtxt("y_m22_2000_18.dat", dtype=np.float32)

x_Mu1= np.loadtxt("X_mu2_2000_18.dat", dtype=np.float32)
x_Lambda1= np.loadtxt("X_lambda2_2000_18.dat", dtype=np.float32)
x_C11= np.loadtxt("X_c12_2000_18.dat", dtype=np.float32)
x_C21= np.loadtxt("X_c22_2000_18.dat", dtype=np.float32)
x_Delta1= np.loadtxt("X_delta2_2000_18.dat", dtype=np.float32)
x_Q1= np.loadtxt("X_q2_2000_18.dat", dtype=np.float32)

x_mu1=np.repeat(x_Mu1[:,:,np.newaxis],200,axis=2)
x_lambda1=np.repeat(x_Lambda1[:,:,np.newaxis],200,axis=2)
x_c11=np.repeat(x_C11[:,:,np.newaxis],200,axis=2)
x_c21=np.repeat(x_C21[:,:,np.newaxis],200,axis=2)
x_delta1=np.repeat(x_Delta1[:,:,np.newaxis],200,axis=2)
AA=np.ones((200,200))
x_q1=x_Q1[:,np.newaxis,np.newaxis]*AA


x_mu1 = np.array(x_mu1, dtype=np.float32)
x_lambda1 = np.array(x_lambda1, dtype=np.float32)
x_c11= np.array(x_c11, dtype=np.float32)
x_c21 = np.array(x_c21, dtype=np.float32)
x_delta1 = np.array(x_delta1, dtype=np.float32)
x_q1 = np.array(x_q1, dtype=np.float32)

x_mu = x_mu1.reshape(x_mu1.shape[0], -1)
x_lambda = x_lambda1.reshape(x_lambda1.shape[0], -1)
x_c1 = x_c11.reshape(x_c11.shape[0], -1)
x_c2 = x_c21.reshape(x_c21.shape[0], -1)
x_delta = x_delta1.reshape(x_delta1.shape[0], -1)
x_q= x_q1.reshape(x_q1.shape[0], -1)

# Create train/test splits
x = np.stack((x_mu,x_lambda,x_c1,x_c2,x_delta,x_q), axis=1)
print(x.shape)
y = y_K

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
print("Data load success!")

def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv2d(6, 16, 5, stride=2) # 4输入
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2)
        self.fc1 = torch.nn.Linear(70688, 1028)
        self.fc2 = torch.nn.Linear(1028, 256)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1, self.shape, self.shape))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define a sequential torch network for batch and trunk. Can use COV2D which we will show later
dim_x = 2
m = (nx)**2

branch = BranchNet(nx)

model = dde.nn.DeepONetCartesianProd([m, branch], [dim_x, 128, 256, 256], "relu", "Glorot normal").cuda()
print("model=", count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss = torch.nn.MSELoss()
train_lossArr = []
test_lossArr = []
time_Arr = []

print("epoch \ttime \t\t\ttrain_loss \t\t\ttest_loss \tlr")
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()
        x = torch.reshape(x, (x.shape[0], 6, nx, nx))

        optimizer.zero_grad()
        out = model((x, grid))
        lp = loss(out, y)
        lp.backward()

        optimizer.step()
        train_loss += lp.item()

    scheduler.step()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in testData:
            x, y = x.cuda(), y.cuda()

            out = model((x, grid))
            test_loss += loss(out, y).item()

    train_loss /= len(trainData)
    test_loss /= len(testData)

    train_lossArr.append(train_loss)
    test_lossArr.append(test_loss)

    t2 = default_timer()
    time_Arr.append(t2-t1)
    if ep%50 == 0:
        print("{} \t{} \t{} \t{} \t{}".format(ep, t2-t1, train_loss, test_loss, 
                                              optimizer.state_dict()['param_groups'][0]['lr']))

# Display Model Details
plt.figure()
plt.plot(train_lossArr, label="Train Loss")
plt.plot(test_lossArr, label="Test Loss")
plt.yscale("log")
plt.legend()
plt.savefig('Train_Test_Loss.png')
plt.show()

testLoss = 0
trainLoss = 0
with torch.no_grad():
    for x, y in trainData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        trainLoss += loss(out, y).item()
        
    for x, y in testData:
        x, y = x.cuda(), y.cuda()

        out = model((x, grid))
        testLoss += loss(out, y).item()
    

# cpu inference
time_Arr_cpu = []
for x, y in testData:
    x, y = x.cpu(), y.cpu()
    model_cpu = model.to("cpu")
    grid_cpu = grid.cpu()

    t1_cpu = default_timer()
    out = model_cpu((x, grid_cpu))
    t2_cpu = default_timer()
    time_Arr_cpu.append(t2_cpu-t1_cpu)


time_Arr_gpu = []
for x, y in testData:
    x, y = x.cuda(), y.cuda()
    model = model.to("cuda:0")

    t1_gpu = default_timer()
    out = model((x, grid))
    t2_gpu = default_timer()
    time_Arr_gpu.append(t2_gpu-t1_gpu)


print("Training Time:", sum(time_Arr))    
print("Avg Epoch Time:", sum(time_Arr)/len(time_Arr))
print("CPU Avg Caculate Time:", sum(time_Arr_cpu)/len(time_Arr_cpu))
print("GPU Avg Caculate Time:", sum(time_Arr_gpu)/len(time_Arr_gpu))
print("Final Testing Loss:", testLoss / len(testData))
print("Final Training Loss:", trainLoss / len(trainData))

torch.save(model, "Hyperbolic_")

# Hyperbolic_k2_2000
# (40000, 2)
# grid.shape= torch.Size([40000, 6])
# (2000, 6, 40000)
# torch.Size([1800, 6, 40000]) torch.Size([200, 6, 40000]) torch.Size([1800, 40000]) torch.Size([200, 40000])
# Data load success!
# model= 73046677
# epoch   time                    train_loss                      test_loss       lr
# 0       1.8441348360211123      305.3038619105263       0.007850181510938066    0.001
# 50      1.9802296989946626      0.00026846249902359487  0.0002636403493982341   0.0005
# 100     1.9815708860114682      0.0001498680886518406   0.00015458817506441846  0.00025
# 150     1.980536959017627       8.2617421229084e-05     9.015044530055352e-05   0.000125
# 200     1.9818642720056232      6.689272547526726e-05   7.038699394407948e-05   6.25e-05
# 250     1.9801349909976125      5.739991460710339e-05   6.196151454267757e-05   3.125e-05
# 300     1.9810639810166322      4.8508197658903605e-05  5.293477131220113e-05   1.5625e-05
# 350     1.9833252819953486      4.413454396917746e-05   4.961255971076233e-05   7.8125e-06
# 400     1.980271605978487       4.109670069801883e-05   4.568754281665731e-05   3.90625e-06
# 450     1.9796310589881614      3.897915405360165e-05   4.353291166937977e-05   1.953125e-06
# 500     1.980392347992165       3.8053337049317184e-05  4.188111363743831e-05   9.765625e-07
# 550     1.981640106998384       3.688159625182794e-05   4.071724262238214e-05   4.8828125e-07
# 600     1.9796764719940256      3.5833718293201456e-05  3.9863598950822574e-05  2.44140625e-07
# 650     1.980006906989729       3.525279945486609e-05   3.9380344138148106e-05  1.220703125e-07
# 700     1.98033022400341        3.5016622413215306e-05  3.914093213097658e-05   6.103515625e-08
# 750     1.9805825320072472      3.487203684926499e-05   3.899469759614606e-05   3.0517578125e-08
# Training Time: 1583.1791178103595
# Avg Epoch Time: 1.9789738972629494
# CPU Avg Caculate Time: 3.4169826814239577
# GPU Avg Caculate Time: 0.002439551575142624
# Final Testing Loss: 3.890703062227528e-05
# Final Training Loss: 3.4781375768688394e-05

# Hyperbolic_k1_2000
# (40000, 2)
# grid.shape= torch.Size([40000, 6])
# (2000, 6, 40000)
# torch.Size([1800, 6, 40000]) torch.Size([200, 6, 40000]) torch.Size([1800, 40000]) torch.Size([200, 40000])
# Data load success!
# model= 73046677
# epoch   time                    train_loss                      test_loss       lr
# 0       1.882676029985305       452.0744807278051       0.01586832824562277     0.001
# 50      1.981920780002838       0.00031140559606188746  0.0003156074339390865   0.0005
# 100     1.982759578997502       0.00022052171187712238  0.00021820161782670766  0.00025
# 150     1.9856906739878468      0.0001373247100753105   0.00014189829068657543  0.000125
# 200     1.9834719510108698      0.00011394734853335346  0.00012553223080301126  6.25e-05
# 250     1.9823378139990382      9.748311174661739e-05   9.955701430694066e-05   3.125e-05
# 300     1.9826439479948021      8.114965914916084e-05   8.608580433896609e-05   1.5625e-05
# 350     2.049825021997094       6.967493754928e-05      7.376260535758255e-05   7.8125e-06
# 400     1.9792676189972553      6.159764208796172e-05   6.500695391358542e-05   3.90625e-06
# 450     1.9796937300125137      5.6358426630603185e-05  6.055677710849393e-05   1.953125e-06
# 500     1.9804412279918324      5.3489708909438105e-05  5.7554350080733586e-05  9.765625e-07
# 550     1.9809591390076093      5.181380427593627e-05   5.5961234465939924e-05  4.8828125e-07
# 600     1.9812628010113258      5.0684997212987294e-05  5.479848031037753e-05   2.44140625e-07
# 650     1.9825067190104164      4.965344972636861e-05   5.393077663029544e-05   1.220703125e-07
# 700     1.9818625510088168      4.935030473892479e-05   5.358960905661141e-05   6.103515625e-08
# 750     1.9832784789905418      4.9181922102498154e-05  5.3376953603999156e-05  3.0517578125e-08
# Training Time: 1585.8767728056118
# Avg Epoch Time: 1.9823459660070148
# CPU Avg Caculate Time: 3.3902805262852262
# GPU Avg Caculate Time: 0.0020648685624889496
# Final Testing Loss: 5.3262886857347826e-05
# Final Training Loss: 4.897616325475259e-05

# Hyperbolic_m1_2000
# (40000, 2)
# grid.shape= torch.Size([40000, 6])
# (2000, 6, 40000)
# torch.Size([1800, 6, 40000]) torch.Size([200, 6, 40000]) torch.Size([1800, 40000]) torch.Size([200, 40000])
# Data load success!
# model= 73046677
# epoch   time                    train_loss                      test_loss       lr
# 0       1.8459814759844448      851.2291863714917       0.04430437034794262     0.001
# 50      1.982943837007042       0.0007709928240888475   0.0007322790500308786   0.0005
# 100     1.9827604389865883      0.00023143957961318002  0.00023422189198234783  0.00025
# 150     1.9821299450122751      0.0001591511980605949   0.00016703630431688258  0.000125
# 200     1.9895408659940585      0.0001634996078792028   0.0001568074055415179   6.25e-05
# 250     1.9872098719934002      0.00012102669853974428  0.00012807250771272396  3.125e-05
# 300     1.982130636984948       0.00010775383602697075  0.00011588819221027993  1.5625e-05
# 350     1.9831736650085077      9.917235346572277e-05   0.00010535877663642168  7.8125e-06
# 400     1.9824503510026261      9.088613577452569e-05   9.87977151193523e-05    3.90625e-06
# 450     1.9846098140114918      8.47238594647742e-05    9.23456931819341e-05    1.953125e-06
# 500     1.9832884219940752      7.953765389015244e-05   8.632426657381334e-05   9.765625e-07
# 550     1.9828381070110481      7.503302983598572e-05   8.174809045158327e-05   4.8828125e-07
# 600     1.9834286580153275      7.171739873643801e-05   7.783628825563937e-05   2.44140625e-07
# 650     1.9827699799789116      6.912994119869709e-05   7.580419540837673e-05   1.220703125e-07
# 700     1.9826967570115812      6.815996024861785e-05   7.448307016082773e-05   6.103515625e-08
# 750     1.9825516730197705      6.739482431153751e-05   7.381592877209187e-05   3.0517578125e-08
# Training Time: 1586.194780429767
# Avg Epoch Time: 1.9827434755372086
# CPU Avg Caculate Time: 3.182069129572483
# GPU Avg Caculate Time: 0.0007817831480809088
# Final Testing Loss: 7.342866956605576e-05
# Final Training Loss: 6.69441157553688e-05

# Hyperbolic_m2_18_2000
# (40000, 2)
# grid.shape= torch.Size([40000, 6])
# (2000, 6, 40000)
# torch.Size([1800, 6, 40000]) torch.Size([200, 6, 40000]) torch.Size([1800, 40000]) torch.Size([200, 40000])
# Data load success!
# model= 73046677
# epoch   time                    train_loss                      test_loss       lr
# 0       1.839723770011915       394.5046378106488       0.0022522280757714596   0.001
# 50      1.979972789005842       0.00015319715812598078  0.0001431070185000343   0.0005
# 100     1.9792296650120988      9.116338192701765e-05   9.759427380881139e-05   0.00025
# 150     1.9783966539835092      7.79633272414733e-05    7.58993233570696e-05    0.000125
# 200     1.9782158279849682      6.316897602039062e-05   6.263715580904059e-05   6.25e-05
# 250     1.9791517530102283      5.295528420614765e-05   5.272507457578156e-05   3.125e-05
# 300     1.9807821300055366      4.6043899808928633e-05  4.597254727351745e-05   1.5625e-05
# 350     1.9782700229843613      4.123382920321698e-05   4.1671902831045114e-05  7.8125e-06
# 400     1.9788402749982197      3.61128113262678e-05    3.5882471950442e-05     3.90625e-06
# 450     1.9793234969838522      3.2896755248912724e-05  3.272556101105043e-05   1.953125e-06
# 500     1.97884551298921        3.0845232686260715e-05  3.059118794876018e-05   9.765625e-07
# 550     1.980145405017538       2.9090891155391818e-05  2.884470632125158e-05   4.8828125e-07
# 600     1.9796454109891783      2.7688508405270496e-05  2.7652709832182154e-05  2.44140625e-07
# 650     1.9789312340144534      2.6921626334099035e-05  2.690194820128714e-05   1.220703125e-07
# 700     1.97914140400826        2.6460081611284087e-05  2.6475736116740984e-05  6.103515625e-08
# 750     1.9792962659848854      2.6272299548604917e-05  2.6260099860207576e-05  3.0517578125e-08
# Training Time: 1582.0734618781134
# Avg Epoch Time: 1.9775918273476418
# CPU Avg Caculate Time: 3.439101839142885
# GPU Avg Caculate Time: 0.001953396713361144
# Final Testing Loss: 2.615504568633956e-05
# Final Training Loss: 2.6165377564387183e-05
