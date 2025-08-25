

from Object_detection_2d.scheduler.poly_scheduler import PolyLR
from Object_detection_2d.scheduler.step_scheduler import StepScheduler
from Object_detection_2d.scheduler.warmup_multi_step_scheduler import WarmupMultiStepLR

SCHEDULER_DICT = {
    "Poly": PolyLR,
    "Step": StepScheduler,
    "WarmupMultiStep": WarmupMultiStepLR
}
 
def build_scheduler(args, optimizer):
    scheduler_config = args['scheduler']
    scheduler_name = scheduler_config.pop('name', None)
    scheduler_factory = SCHEDULER_DICT[scheduler_name]
    return scheduler_factory(optimizer, **scheduler_config)