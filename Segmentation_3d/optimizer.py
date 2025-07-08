from torch.optim.lr_scheduler import _LRScheduler, StepLR, ExponentialLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]
        
def get_scheduler(args, optimizer):
    if args.scheduler == "poly":
        return PolyLR(optimizer, args.epochs)
    elif args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.scheduler == "exp":
        return ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError(f'Unknown scheduler {args.scheduler}')