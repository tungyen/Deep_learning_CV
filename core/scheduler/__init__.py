from core.scheduler.poly_scheduler import PolyLR
from core.scheduler.step_scheduler import StepScheduler
from core.scheduler.warmup_multi_step_scheduler import WarmupMultiStepLR
from core.scheduler.warmup_cosine_annealing_scheduler import CosineAnnealingWarmup

from torch.optim.lr_scheduler import _LRScheduler, StepLR, ExponentialLR

SCHEDULER_DICT = {
    "Poly": PolyLR,
    "Step": StepScheduler,
    "WarmupMultiStep": WarmupMultiStepLR,
    "CosineAnnealingWarmup": CosineAnnealingWarmup,
    "Exponential": ExponentialLR,
}
 
def build_scheduler(opts, optimizer):
    scheduler_name = opts.pop('name', None)
    if scheduler_name is None or scheduler_name not in SCHEDULER_DICT:
        raise ValueError(f"Missing scheduler name or unknown scheduler {scheduler_name}.")
    scheduler_factory = SCHEDULER_DICT[scheduler_name]
    return scheduler_factory(optimizer, **opts)