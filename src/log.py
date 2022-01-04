class AverageMeter(object):
    """Compute and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_av_meters(av_meters, meters, sizes):
    dif_size = sizes['dif_size']
    sim_size = sizes['dif_size']
    total_size = sizes['total_size']

    if dif_size != 0:
        av_meters['dif_loss'].update(meters['dif_loss'].item(), dif_size)
        av_meters['dif_acc'].update(meters['dif_acc'].item(), dif_size)

    if sim_size != 0:
        av_meters['sim_loss'].update(meters['sim_loss'].item(), sim_size)
        av_meters['sim_acc'].update(meters['sim_acc'].item(), sim_size)

    av_meters['total_loss'].update(meters['total_loss'].item(), total_size)
    av_meters['total_acc'].update(meters['total_acc'].item(), total_size)


def update_writers(writer, av_meters, train_or_test, epoch):

    def add(key, value):
        writer.add_scalar(key, value, epoch)

    add(f'{train_or_test}/total_acc', av_meters['total_acc'].avg)
    add(f'{train_or_test}/dif_acc', av_meters['dif_acc'].avg)
    add(f'{train_or_test}/sim_acc', av_meters['sim_acc'].avg)
    add(f'{train_or_test}/total_loss', av_meters['total_loss'].avg)
    add(f'{train_or_test}/dif_loss', av_meters['dif_loss'].avg)
    add(f'{train_or_test}/sim_loss', av_meters['sim_loss'].avg)
