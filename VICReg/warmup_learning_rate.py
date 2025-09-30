import tensorflow as tf
import numpy as np


class WarmUpLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, lr_func, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpLR, self).__init__()

        self.lr_func              = lr_func
        self.learning_rate_base   = learning_rate_base
        self.total_steps          = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps         = warmup_steps
        self.pi                   = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (self.learning_rate_base*lr_func(tf.cast(step, tf.float32)-self.warmup_steps)/float(self.total_steps - self.warmup_steps))/lr_func(0)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(step < self.warmup_steps, warmup_rate, learning_rate)
        return tf.where(step > self.total_steps, 0.0, learning_rate, name="learning_rate")
