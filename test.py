import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List
from torch import nn, Tensor
labels_in=torch.Tensor([1,5,8])
best_dbox_ious=torch.Tensor([2,2,2,0.3,0.6])
best_dbox_idx=torch.Tensor([0,2,1,2,2])
masks=torch.Tensor([True,True,True,False,True]).type(torch.bool)
labels_out = torch.Tensor([0,0, 0,0,0])
print(best_dbox_idx)
print(masks)
print(best_dbox_idx[masks])
labels_out[masks] = labels_in[best_dbox_idx[masks].type(torch.long)]
print(labels_out)