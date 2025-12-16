import torch

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.val} ({self.avg})'

@torch.no_grad()
def accuracy(output, target):
    # output: (batch, num_classes), target: (batch)
    preds = torch.argmax(output, dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)