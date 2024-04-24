import torch

from collections import OrderedDict


class StatisticsSaver(object):
    def __init__(self):

        # the length of the bellow dictionaries is assumed to be the batch size

        """
        key:epoch nubmer
        value:list of losses
        """
        self.discriminator_loss_real = OrderedDict()

        """
        key:epoch nubmer
        value:number of batches
        """
        self.number_batches = OrderedDict()

        self.mlp_model = None
        self.mlp_optimizer = None

        self.losses_total = []
        self.accuracy_total = []

        self.total_traininng_time = []

    def set_trainned_model(self, mlp_model):
        self.mlp_model = mlp_model

    def set_trainned_optimizer(self, mlp_optimizer):
        self.mlp_optimizer = mlp_optimizer

    def save_trainned_model(self, save_model_file_name, last_epoch):

        torch.save(
            {
                "epoch": last_epoch,
                "model_generator": self.mlp_model,
                "optimizer_generator": self.mlp_optimizer,
                "loss": self,
            },
            save_model_file_name + "_whole" + ".pth",
        )

    def save_model_params(self, save_model_file_name, last_epoch):

        torch.save(
            {
                "epoch": last_epoch,
                "model_state_dict": self.mlp_model.state_dict(),
                "optimizer_state_dict": self.mlp_optimizer.state_dict(),
            },
            save_model_file_name + "_params" + ".pth",
        )

    def save_losses(self, losses):
        self.losses_total.append(losses)

    def save_accuracy(self, accuracy):
        self.accuracy_total.append(accuracy)

    def save_total_training_time(self, total_time):
        self.total_traininng_time.append(total_time)
