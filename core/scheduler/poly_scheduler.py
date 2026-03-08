from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, power=0.9, min_lr=1e-6, last_epoch=-1, train_size=-1, epochs=1, **kwargs):
        self.power = power
        self.max_iters = train_size * epochs
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]