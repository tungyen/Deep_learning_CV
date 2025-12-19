from torch.optim.lr_scheduler import StepLR

class StepScheduler(StepLR):
    def __init__(self, optimizer, step_size, gamma):
        step_size = step_size
        gamma = gamma
        super(StepScheduler, self).__init__(optimizer, step_size, gamma)