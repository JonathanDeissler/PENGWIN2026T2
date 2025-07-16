from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class PytorchCompliantPolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, last_epoch: int = -1):
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch, self.max_steps)
        factor = (1 - step / self.max_steps) ** self.exponent
        return [self.initial_lr * factor for _ in self.optimizer.param_groups]