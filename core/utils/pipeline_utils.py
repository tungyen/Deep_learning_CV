import torch

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, local_rank):
    map_location = {'cuda:0': f'cuda:{local_rank}'}
    checkpoint = torch.load(path, map_location=map_location)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_metric = checkpoint['best_metric']
    return start_epoch, best_metric