import tensorflow as tf
from tensorflow import keras
import transformers


"""
This file contains a standard prediction head (Dense --> Softmax over vocab elements)
as well as a "Split Prediction" option. The split prediction
"""

class SplitPred(keras.layers.Layer):

    def __init__(self, vocab_size, dual_heads_split_step, name='SplitPred', **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.dual_heads_split_step = dual_heads_split_step

        self.Dense_0 = keras.layers.Dense(self.vocab_size, name='Dense_0')
        self.Dense_1 = keras.layers.Dense(self.vocab_size, name='Dense_1')

        self.Split = keras.layers.Lambda(lambda x: [x[:, :self.dual_heads_split_step, :], 
                                                   x[:, self.dual_heads_split_step:, :]],
                                         name='Split')

        self.Rejoin = keras.layers.Concatenate(axis=1, name='Rejoin')                                                   
    
    def config(self):
        config = super().get_config()
        config.update({'vocab_size':self.vocab_size, 'dual_heads_split_step':self.dual_heads_split_step})
        return config

    def call(self, inputs):
        features = inputs
        features_0, features_1 = self.Split(features)

        features_0 = self.Dense_0(features_0)
        features_1 = self.Dense_1(features_1)

        logits = self.Rejoin([features_0, features_1])
        return logits


class DecoderHead(keras.layers.Layer):

    def __init__(self, vocab_size, dual_heads_split_step=None, name='DecoderHead', **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.dual_heads_split_step = dual_heads_split_step
        self.Softmax = keras.layers.Softmax(dtype=tf.float32)

    def build(self, input_shape):
        self.features_shape = input_shape

        if self.dual_heads_split_step:
            self.Dense = SplitPred(self.vocab_size, self.dual_heads_split_step)
        else:
            self.Dense = keras.layers.Dense(self.vocab_size)


    def config(self):
        config = super().get_config()
        config.update({'vocab_size':self.vocab_size, 'dual_heads_split_step':self.dual_heads_split_step})
        return config
    
    def call(self, inputs):

        features = inputs
        logits = self.Dense(features)     
        probabilities = self.Softmax(logits)

        return probabilities

    def show_summary(self):    
        features = keras.layers.Input(shape=self.features_shape[1:], name='features')
        inputs = features
        return tf.keras.Model(inputs, outputs=self.call(inputs), name=self.name).summary()



class BeamUpdate(keras.layers.Layer):

    def __init__(self, num_attention_heads, vocab_size, name='BeamUpdate', **kwargs):
        super().__init__(name=name, **kwargs) 

        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size

        # layers
        self.BeamUnit = keras.layers.GRU(self.vocab_size, return_sequences=True, 
                            return_state=False, go_backwards=True, dtype=tf.float32,
                            name='BeamUnit')  

        self.SelfAttention = transformers.AttentionBlock(num_attention_heads=self.num_attention_heads, 
                                                     name='SelfAttention')
        self.JointAttention = transformers.AttentionBlock(num_attention_heads=self.num_attention_heads, 
                                                     name='JointAttention')

        self.FeedForwardLayer = transformers.FeedForwardBlock(name='FeedForwardLayer')
        self.DecoderHead = DecoderHead(vocab_size=self.vocab_size,  dual_heads_split_step=None)

    def build(self, input_shapes):
        self.predictions_shape = input_shapes[0]
        self.encoder_features_shape = input_shapes[1]


    def call(self, inputs):

        predictions = inputs[0]  # [batch, num steps, vocab size]
        encoder_features = inputs[1]  # [batch, num_vectors, encoder_dim]
                
        # pass original predictions through RNN
        predictions = self.BeamUnit(predictions)

        # attention update (no masking)
        predictions = self.SelfAttention([predictions, predictions, predictions])
        predictions = self.JointAttention([predictions, encoder_features, encoder_features])
        predictions = self.FeedForwardLayer(predictions)
        
        # final prediction
        probs = self.DecoderHead(predictions)  

        return probs

    def show_summary(self):
        orig_predictions = tf.keras.layers.Input(self.predictions_shape[1:], name='orig_predictions')
        encoder_features = tf.keras.layers.Input(self.encoder_features_shape[1:], name='encoder_features')
        inputs = [orig_predictions, encoder_features]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()