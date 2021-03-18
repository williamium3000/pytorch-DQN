import numpy as np
import torch

t = torch.tensor([[1, 4, 5], [2, 3, 7], [4, 6, 9]])
print(np.argmax(t, axis=1))
