from torch.optim.lr_scheduler import StepLR

class StepScheduler(StepLR):
    def __init__(self, optimizer, args):
        step_size = args['step_size']
        gamma = args['gamma']
        super(StepScheduler, self).__init__(optimizer, step_size, gamma)