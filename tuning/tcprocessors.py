# OM NAMO GANAPATHAYEN NAMAHA
from transformers import TrainerState
import numpy as np
from collections import deque

class TrainerControlProcessors:
    def compute(self, state: TrainerState):
        pass

    def get_result(self):
        pass

class AverageOverLossWindow(TrainerControlProcessors):
    def __init__(self, cfg):
        self.sample_window = int(cfg['sample_window'])
        self.avg_loss = 0
        self.cache = deque()

    def compute(self, state: TrainerState):
        log_history = state.log_history
        n = len(log_history)
        if n > 0 and 'loss' in log_history[n-1]:
            self.cache.append(log_history[n-1]['loss'])
            numSamples = len(self.cache)
            if numSamples <= self.sample_window:
                window = np.array(self.cache)
                self.avg_loss = np.mean(window)
            else:
                self.cache.popleft()
                window = np.array(self.cache)
                self.avg_loss = np.mean(window)

    def get_result(self):
        return {'avg_loss': self.avg_loss}

class CompareEpochAverageLoss(TrainerControlProcessors):
    def __init__(self):
        self.cache = deque()

    def compute(self, state: TrainerState):
        log_history = state.log_history
        n = len(log_history)
        x = []
        for i in range(n):
            l = log_history[i]
            if 'loss' in l:
                x.append(l['loss'])
        loss_array = np.array(x)
        avg_loss = np.mean(loss_array)
        self.cache.append(avg_loss)
        if len(self.cache) > 2:
            self.cache.popleft()

    def get_result(self):
        if len(self.cache) < 2:
            return None
        n1 = self.cache.popleft()
        n2 = self.cache.popleft()
        self.cache.append(n1)
        self.cache.append(n2)
        return {'avg_loss[n]': n1, 'avg_loss[n+1]': n2}