import torch
import torch.nn.functional as F

# 定义输入样本数值
X1 = torch.tensor(1)
X2 = torch.tensor(2)
Y = torch.tensor(2)
Lr = 0.01

# 参数初始化
w1_11 = torch.tensor(1.,requires_grad=True)
w1_12 = torch.tensor(1.,requires_grad=True)
w1_21 = torch.tensor(1.,requires_grad=True)
w1_22 = torch.tensor(-1.,requires_grad=True)
w2_11 = torch.tensor(1.,requires_grad=True)
w2_21 = torch.tensor(-1.,requires_grad=True)

𝜽1_11, 𝜽1_12, 𝜽2_11 = torch.tensor([1.,1.,1.],requires_grad=True)

# 前向传播
y1_1 = w1_11*X1 + w1_21*X2 + 𝜽1_11
x2_1 = F.relu(y1_1)
y1_2 = w1_12*X1 + w1_22*X2 + 𝜽1_12
x2_2 = F.relu(y1_2)
A = w2_11*x2_1 + w2_21*x2_2 +𝜽2_11

# Loss
Loss = (A-Y).pow(2)/2

Loss.backward()
print(w1_11.grad)

with torch.no_grad():
    w1_11 -= Lr*w1_11.grad

print(w1_11)
