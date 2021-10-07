import tensorflow as tf
from tensorflow import keras


"""
EditDistanceMetric measures the Levenstein distance between two lists. Here we 
apply it to our tokenized InChi predictions. Note that our tokenization consolidates
some text (sor example 'Si' is a single token), so that this differs from the 
Lev distance on the text strings.

Note: This function is not TPU-compatible.
Note: This function is not differentiable, and cannot be used as a loss function
"""
class EditDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='EditDistanceMetric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.edit_distance = self.add_weight(name='edit_distance', initializer='zeros')
        self.batch_counter = self.add_weight(name='batch_counter', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)  # convert one_hot vectors back to sparce categorical
        y_true = tf.sparse.from_dense(y_true)
        y_pred = tf.sparse.from_dense(tf.argmax(y_pred, axis=-1))  # convert probs to preds

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # compute edit distance (of parsed tokens)
        edit_distance = tf.edit_distance(y_pred, y_true, normalize=False)
        self.edit_distance.assign_add(tf.reduce_mean(edit_distance))

        # update counter
        self.batch_counter.assign_add(tf.reduce_sum(1.))
    
    def result(self):
        return self.edit_distance / self.batch_counter

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.edit_distance.assign(0.0)
        self.batch_counter.assign(0.0)


"""
A modified version of the learning rate scheduler described in 
"Attention is All You Need."  This is a cyclic version of their scheduler.
"""
class LRScheduleAIAYN(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, scale_factor=1, warmup_steps=4000):  # defaults reflect paper's values
        # cast dtypes
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        dim = tf.constant(352, dtype=tf.float32)
        scale_factor = tf.constant(scale_factor, dtype=tf.float32)
        
        self.scale = scale_factor * tf.math.pow(dim, -1.5)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        crit = self.warmup_steps

        def false_fn(step):
            adj_step = (step - crit) % (2.0*crit) + crit
            return tf.math.pow(adj_step, -.5)

        val = tf.cond(tf.math.less(step, crit),
                      lambda: step * tf.math.pow(crit, -1.5),  # linear increase
                      lambda: false_fn(step)  # decay
                      )

        return self.scale * val

    def visualize(self):
        # plots a graph of the learning rate 
        plt.plot([i for i in range(1, 16000)], [self(i) for i in range(1, 16000)])
        plt.show()
        print('Learning Rate Schedule')



