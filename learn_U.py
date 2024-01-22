import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from sklearn.model_selection import train_test_split
import operator
from timeit import default_timer
from matplotlib.ticker import FormatStrFormatter
import deepxde as dde
import pdb

# Parameters
epochs =800
ntrain = 1800
ntest = 200
batch_size = 32
gamma = 0.5
learning_rate = 0.001
step_size= 50
modes=12
width=32


X = 1
dx = 1/201
nx = 201
Al=np.zeros((1,nx+1))
Al[0, 0]=1/3
Al[0, nx]=1/3
for i in range(1, nx):
    if i % 2==0:
        Al[0, i]=4/3
    else:
        Al[0, i]=2/3
Al=Al*dx

grid = np.linspace(0, X, nx+1, dtype=np.float32).reshape(nx+1, 1)

u=np.zeros(1000)
Y_500=np.loadtxt("Y_500.dat", dtype=np.float32)
Y_1500=np.loadtxt("Y_1500.dat", dtype=np.float32)
u0=np.zeros((500,1))
u00=np.zeros((1500,1))
for i in range(500):
    u0[i]= np.matmul(Y_500[i,:],Al.reshape(202,1))

for i in range(1500):
    u00[i]= np.matmul(Y_1500[i,:],Al.reshape(202,1))

pdb.set_trace()
 
y=np.zeros((2000,202))
u=np.zeros((2000,1))
y[0:500,:]=Y_500
y[500:2000,:]=Y_1500
u[0:500]=u0
u[500:2000]=u00
print(y.shape,u.shape)
pdb.set_trace()
u = np.array(u, dtype=np.float32)
y = np.array(y, dtype=np.float32)

x_train, x_test, y_train, y_test = train_test_split(y, u, test_size=0.1, random_state=1)
x_train = torch.from_numpy(x_train).cuda()
y_train = torch.from_numpy(y_train).cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).cuda()
grid = torch.from_numpy(np.array(grid, dtype=np.float32)).cuda()

trainData = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
testData = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

# Define a sequential torch network for batch and trunk. Can use COV2D which we will show in 2D problem
dim_x = 1
m = nx+1
branch = [m, 256, 256]
trunk = [dim_x, 128, 256]
activation = "relu"
kernel = "Glorot normal"

def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class DeepONetModified(nn.Module):
    def __init__(self, branch, trunk, activation, kernel, projection):
        super(DeepONetModified, self).__init__()
        self.net1 = dde.nn.DeepONetCartesianProd(branch, trunk, activation, kernel).cuda()
        self.fc1 = nn.Linear(m, m)
        self.fc2 = nn.Linear(m, m)
        self.net2 = dde.nn.DeepONetCartesianProd(branch, trunk, activation, kernel).cuda()
        self.fc3 = nn.Linear(m,128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, projection)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x, grid = x[0], x[1]
        x = self.net1((x, grid))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.net2((x, grid))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
       
projection  = 1
model = DeepONetModified(branch, trunk, activation, kernel, projection)
print("model paramater: ", count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

loss = nn.MSELoss()
train_lossArr = []
test_lossArr = []
time_Arr = []

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_loss = 0
    for x, y in trainData:
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
            # print(x.shape, y.shape)
            out = model((x, grid))
            test_loss += loss(out, y).item()
            
    train_loss /= len(trainData)
    test_loss /= len(testData)
    
    train_lossArr.append(train_loss)
    test_lossArr.append(test_loss)
    
    t2 = default_timer()
    time_Arr.append(t2-t1)
    if ep%50 == 0:
        print(ep, t2-t1, np.mean(train_lossArr[-50:]), np.mean(test_lossArr[-50:]))

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

torch.save(model, "U_1000")
# U_1000
# model paramater:  418399
# 0 0.30772417405387387 0.1227875948626254 0.0006849144618692142
# 50 0.200275128998328 0.0008876905571162896 0.0005516063988047141
# 100 0.20787688996642828 0.00011254818356862592 6.309171023921019e-05
# 150 0.21115560695761815 4.903542676209669e-05 5.654766927152974e-05
# 200 0.208922226971481 1.4296364090913163e-05 1.617436761843497e-05
# 250 0.20748416299466044 2.44694263114441e-06 5.538085991523595e-06
# 300 0.20988293300615624 1.3543659380270144e-06 1.4084539844816393e-06
# 350 0.2110326699912548 4.4356464711731846e-07 4.7244985958221666e-07
# 400 0.2111259059747681 2.180061835638139e-07 2.97844632487657e-07
# 450 0.20810447598341852 1.2206032724520936e-07 1.416443023671832e-07
# 500 0.20040207204874605 9.208894550183757e-08 1.1254741514120171e-07
# 550 0.2098981780000031 7.236591764079392e-08 8.116222779059563e-08
# 600 0.20925477700075135 6.410746285881598e-08 7.096531967779437e-08
# 650 0.2027443940169178 5.897316159159604e-08 6.623413269782727e-08
# 700 0.20752179896226153 5.6252029470483854e-08 6.273488189886426e-08
# 750 0.21048861800227314 5.519502465620406e-08 6.1860287586971e-08
# Training Time: 167.1645664446405
# Avg Epoch Time: 0.20895570805580063
# CPU Avg Caculate Time: 0.022030987444200685
# GPU Avg Caculate Time: 0.0006839439988003246
# Final Testing Loss: 5.972136766071604e-08
# Final Training Loss: 5.4569573997921733e-08