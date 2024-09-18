import torch

print(torch.cuda.is_available())

# 간단한 텐서 생성
data_1 = [[1, 2], [3, 4]]
data_2 = torch.tensor(data_1)
print(data_2)