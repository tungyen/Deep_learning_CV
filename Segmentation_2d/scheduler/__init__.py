from Segmentation_2d.scheduler.poly_scheduler import PolyLR
from Segmentation_2d.scheduler.step_scheduler import StepScheduler
from Segmentation_2d.scheduler.warmup_multi_step_scheduler import WarmupMultiStepLR
from Segmentation_2d.scheduler.warmup_cosine_annealing_scheduler import CosineAnnealingWarmup

SCHEDULER_DICT = {
    "Poly": PolyLR,
    "Step": StepScheduler,
    "WarmupMultiStep": WarmupMultiStepLR,
    "CosineAnnealingWarmup": CosineAnnealingWarmup,
}
 
def build_scheduler(opts, optimizer):
    scheduler_name = opts.pop('name', None)
    if scheduler_name is None or scheduler_name not in SCHEDULER_DICT:
        raise ValueError(f"Missing scheduler name or unknown scheduler {scheduler_name}.")
    scheduler_factory = SCHEDULER_DICT[scheduler_name]
    return scheduler_factory(optimizer, **opts)