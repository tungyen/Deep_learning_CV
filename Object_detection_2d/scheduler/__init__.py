from Object_detection_2d.scheduler.poly_scheduler import PolyLR
from Object_detection_2d.scheduler.step_scheduler import StepScheduler
from Object_detection_2d.scheduler.warmup_multi_step_scheduler import WarmupMultiStepLR
from Object_detection_2d.scheduler.warmup_cosine_annealing_scheduler import CosineAnnealingWarmup
SCHEDULER_DICT = {
    "Poly": PolyLR,
    "Step": StepScheduler,
    "WarmupMultiStep": WarmupMultiStepLR,
    "CosineAnnealingWarmup": CosineAnnealingWarmup,
}
 
def build_scheduler(args, optimizer):
    scheduler_config = args['scheduler']
    scheduler_name = scheduler_config.pop('name', None)
    if scheduler_name is None or scheduler_name not in SCHEDULER_DICT:
        raise ValueError(f"Missing scheduler name or unknown scheduler {scheduler_name}.")
    scheduler_factory = SCHEDULER_DICT[scheduler_name]
    return scheduler_factory(optimizer, **scheduler_config)