import torch
from torch import nn
def InformationEntropy(pred, target):
    entropy = target * torch.log(pred + 1e-8)
    return torch.mean(-torch.sum(entropy, dim=-1))

# logits = torch.tensor([[0.3,0.5,0.2],[0.2,0.2,0.6]]).view(-1,3)
# pred = torch.softmax(logits, dim=-1)
# target = torch.tensor([[0.,1.,0.],[1.,0.,0.]])
# print(InformationEntropy(pred, target))
# criterion = nn.CrossEntropyLoss()
# print(criterion(logits, torch.LongTensor([1,0])))