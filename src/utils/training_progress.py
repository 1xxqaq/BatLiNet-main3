"""Line-oriented training progress suitable for redirected/nohup logs."""

import math
import sys
import time
from datetime import datetime, timedelta, timezone


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '估算中'
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class TrainingProgress:
    """Estimate this seed's remaining time from its elapsed training work."""

    def __init__(self, model_name, seed, task='未指定', training_number=1,
                 total_trainings=1, log_interval=30.0, clock=time.perf_counter,
                 stream=None):
        if not math.isfinite(log_interval) or log_interval <= 0:
            raise ValueError('log_interval must be finite and positive.')
        if not 1 <= training_number <= total_trainings:
            raise ValueError('Invalid training number/total_trainings.')
        self.model_name = model_name
        self.seed = seed
        self.task = task
        self.training_number = training_number
        self.total_trainings = total_trainings
        self.log_interval = log_interval
        self.clock = clock
        self.stream = stream if stream is not None else sys.stdout
        self.epoch = self.batch = self.completed_batches = 0
        self.total_epochs = self.batches_per_epoch = 0
        self.started_at = self.last_logged_at = None

    def start(self, total_epochs, batches_per_epoch):
        if total_epochs < 1 or batches_per_epoch < 1:
            raise ValueError('Training requires positive epochs and batches.')
        self.total_epochs = total_epochs
        self.batches_per_epoch = batches_per_epoch
        self.started_at = self.clock()
        self.last_logged_at = self.started_at
        self._write('开始训练')

    def epoch_started(self, epoch):
        self.epoch, self.batch = epoch, 0
        if epoch == 1:
            self._write('训练中')

    def batch_finished(self, epoch, batch):
        self.epoch, self.batch = epoch, batch
        self.completed_batches = (epoch - 1) * self.batches_per_epoch + batch
        if (self.completed_batches == 1
                or self.clock() - self.last_logged_at >= self.log_interval):
            self._write('训练中')

    def epoch_finished(self, epoch):
        self.epoch, self.batch = epoch, self.batches_per_epoch
        self.completed_batches = epoch * self.batches_per_epoch
        self._write('本轮完成')

    def phase(self, name):
        self._write(name)

    def finish(self):
        self._write('训练完成')

    def fail(self):
        if self.started_at is not None:
            self._write('训练异常退出')

    def _write(self, phase):
        now = self.clock()
        elapsed = now - self.started_at
        total_batches = self.total_epochs * self.batches_per_epoch
        remaining = None
        if self.completed_batches:
            remaining = (elapsed / self.completed_batches
                         * max(0, total_batches - self.completed_batches))
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime(
            '%Y-%m-%d %H:%M:%S +08:00')
        print(
            f'[{timestamp}] [进度] 模型={self.model_name} | 任务={self.task} | '
            f'种子={self.seed} | 训练={self.training_number}/{self.total_trainings} | '
            f'轮次={self.epoch}/{self.total_epochs} | '
            f'本轮批次={self.batch}/{self.batches_per_epoch} | 阶段={phase} | '
            f'已耗时={format_duration(elapsed)} | '
            f'当前训练预计剩余={format_duration(remaining)} | '
            f'后续训练={self.total_trainings - self.training_number}次',
            file=self.stream, flush=True)
        self.last_logged_at = now
