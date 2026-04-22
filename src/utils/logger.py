from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, data: dict, step: int):
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()