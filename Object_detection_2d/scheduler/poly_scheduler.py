from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, args):
        self.power = args['power']
        self.max_iters = args['max_iters']
        self.min_lr = args['min_lr']
        super(PolyLR, self).__init__(optimizer, args['last_epoch'])
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]