import os.path as osp
from common.ops import LoggerX

class BasicTask(object):
    def __init__(self, opt):
        self.opt = opt
        save_root = getattr(opt, 'save_root', None) or '../output'
        self.logger = LoggerX(save_root=save_root)

    def set_loader(self):
        pass

    def set_model(self):
        pass

    def validate(self, n_iter):
        pass

    def adjust_learning_rate(self, step):
        pass

    def train(self, inputs, n_iter):
        pass
