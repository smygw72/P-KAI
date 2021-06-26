from omegaconf import DictConfig, ListConfig
import mlflow
from mlflow.tracking import MlflowClient


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


class MlflowWriter():
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(
                experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            mlflow.pytorch.log_state_dict(
                model.to('cpu').state_dict(), 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, epoch):
        self.client.log_metric(self.run_id, key, value, epoch)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)


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


def update_writers(tb_writer, ml_writer, av_meters, train_or_test, epoch):

    def add(key, value):
        tb_writer.add_scalar(key, value, epoch)
        ml_writer.log_metric(key, value, epoch)

    # add(f'{train_or_test}/total_acc', av_meters['total_acc'].avg)
    add(f'{train_or_test}/dif_acc', av_meters['dif_acc'].avg)
    # add(f'{train_or_test}/sim_acc', av_meters['sim_acc'].avg)
    add(f'{train_or_test}/total_loss', av_meters['total_loss'].avg)
    add(f'{train_or_test}/dif_loss', av_meters['dif_loss'].avg)
    add(f'{train_or_test}/sim_loss', av_meters['sim_loss'].avg)
