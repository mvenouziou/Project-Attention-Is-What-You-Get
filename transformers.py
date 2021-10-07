import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
This file contains all the attention transformer components described in the paper 
"Attention is All You Need." 

The encoder is comprised of blocks of the form 
Self Attention -> Res Connection --> Feed Forward --> Res Connection
The decoder adds Joint Attention --> Res Connection after each self attention + res
block.  Both encoder and decoder contain trainable positional encodings
"""

class AddPositional(tf.keras.layers.Layer):
    """ Creates and adds a positional encoding variable of shape [batch, num, dim].
    Initialized at value matching AIAYN paper, with option to be trainable. """

    def __init__(self, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.trainable = trainable

    def get_config(self):
        config = super().get_config()
        config.update({'trainable': self.trainable})
        return config

    def build(self, input_shape):

        num_obj = input_shape[1]
        feature_dim = input_shape[2]

        # create positional encoding vector appropriate for 2D vectors
        def trig(k, dim):
            denom_exp = 2*dim / feature_dim
            denom = 10000 ** denom_exp
            even = k % 2
            odd = (k+1) % 2
            return even * np.math.sin(k / denom) + odd * np.math.cos(k / denom)
        self.positional_encoding = tf.Variable([[trig(k, dim) 
                                                 for dim in range(feature_dim)]
                                                 for k in range(num_obj)],
                                                trainable=self.trainable, 
                                                name='positional_encoding')

    def call(self, inputs):
        feature = inputs
        feature = feature + tf.cast(tf.expand_dims(self.positional_encoding, axis=0), 
                                    dtype=feature.dtype)
        return feature


class AttentionBlock(keras.layers.Layer):
    """ Attention Layer --> Residual Addition --> LayerNormalization """
    def __init__(self, num_attention_heads, **kwargs):
        super().__init__(**kwargs)

        self.num_attention_heads = num_attention_heads

    def get_config(self):
        config = super().get_config()
        config.update({'num_attention_heads': self.num_attention_heads})
        return config

    def build(self, input_shape):
        self.query_shape = input_shape[0]
        self.key_shape = input_shape[1]
        self.value_shape = input_shape[2]

        query_dim = self.query_shape[-1]
        key_dim = tf.math.maximum(1, query_dim//self.num_attention_heads)

        self.AttentionLayer = tf.keras.layers.MultiHeadAttention(
              num_heads=self.num_attention_heads, key_dim=key_dim,
              dropout=0.1, name='AttentionLayer')

        self.Add = tf.keras.layers.Add(name='Add')
        self.LayerNorm = tf.keras.layers.LayerNormalization(name='LayerNorm')

    def call(self, inputs, attention_mask=None, training=False):
        query, key, value = inputs

        attention_features = self.AttentionLayer(query=query, 
                                                 value=value,
                                                 key=key,
                                                 attention_mask=attention_mask,
                                                 training=training)      
         
        query = self.Add([query, attention_features])
        query = self.LayerNorm(query, training=training)

        return query

    def show_summary(self):
        query = tf.keras.layers.Input(self.query_shape[1:], name='query')
        key = tf.keras.layers.Input(self.key_shape[1:], name='key')
        value = tf.keras.layers.Input(self.value_shape[1:], name='value')
        inputs=[query, key, value]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()


class FeedForwardBlock(tf.keras.layers.Layer):
    """ Dense Relu --> Dense Linear --> LayerNormalization """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def build(self, input_shape):
        self.features_shape = input_shape
        features_dim = self.features_shape[-1]

        # feed forward
        self.DenseRelu = tf.keras.layers.Dense(features_dim, activation='relu', name='DenseRelu')
        self.DenseLinear = tf.keras.layers.Dense(features_dim, activation=None, name='DenseLinear')
        self.DenseAdd = tf.keras.layers.Add(name='DenseAdd')
        self.DenseLayerNorm = tf.keras.layers.LayerNormalization(name='DenseLayerNorm')
        self.Dropout = tf.keras.layers.Dropout(rate=.01, name='Dropout')
              
    def call(self, inputs, training=False):
        features = inputs

        # Feed Forward block
        dense_features = self.DenseRelu(features)
        dense_features = self.DenseLinear(dense_features)

        dense_features = self.Dropout(dense_features, training=training)
        features = self.DenseAdd([features, dense_features])
        features = self.DenseLayerNorm(features, training=training)

        return features

    def show_summary(self):
        features = tf.keras.layers.Input(self.features_shape[1:], name='features')
        inputs = features
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs)).summary()


class EncoderAttention(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, num_blocks, name='EncoderAttention', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_attention_heads = num_attention_heads
        self.num_blocks = num_blocks

        self.AddPositional = AddPositional(name='AddPositional', trainable=True)
        
        # Attention Blocks
        self.SelfAttentionBlocks = []
        self.FeedForwardBlocks = []

        for i in range(self.num_blocks):
            att = AttentionBlock(self.num_attention_heads, name=f'SelfAttention_{i}')
            feed = FeedForwardBlock(name=f'FeedForward_{i}')

            self.SelfAttentionBlocks.append(att)
            self.FeedForwardBlocks.append(feed)

    def build(self, input_shapes):
        self.Flatten2D = keras.layers.Reshape([-1, input_shapes[-1]], name='Flatten2D')

    def config(self):
        config = super().get_config()
        config.update({'num_attention_heads':self.num_attention_heads, 'num_blocks':self.num_blocks})
        return config

    def call(self, inputs, training=False):
        encoder_features = inputs

        # reshape and add in positional encoding
        encoder_features = self.Flatten2D(encoder_features)
        encoder_features = self.AddPositional(encoder_features)

        # run theough the attention and feed-forward blocks
        for att_layer, feed_layer in zip(self.SelfAttentionBlocks, self.FeedForwardBlocks):

            encoder_features = att_layer([encoder_features, encoder_features, encoder_features],
                                         attention_mask=None, training=training)
            encoder_features = feed_layer(encoder_features, training=training)

        return encoder_features

    def show_summary(self):    
        encoding = keras.layers.Input(shape=[10,10,256])
        inputs = encoding
        return tf.keras.Model(inputs, outputs=self.call(inputs), name=self.name).summary()


class DecoderAttention(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, num_blocks, name='DecoderAttention', **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_attention_heads = num_attention_heads
        self.num_blocks = num_blocks

        self.AddPositional = AddPositional(name='AddPositional', trainable=False)
        
        # Attention Blocks
        self.SelfAttentionBlocks = []
        self.JointAttentionBlocks = []
        self.FeedForwardBlocks = []

        for i in range(self.num_blocks):
            self_att = AttentionBlock(self.num_attention_heads, name=f'SelfAttention_{i}')
            joint_att = AttentionBlock(self.num_attention_heads, name=f'JointAttention_{i}')
            feed = FeedForwardBlock(name=f'FeedForward_{i}')

            self.SelfAttentionBlocks.append(self_att)
            self.JointAttentionBlocks.append(joint_att)
            self.FeedForwardBlocks.append(feed)

    def config(self):
        config = super().get_config()
        config.update({'num_attention_heads':self.num_attention_heads, 'num_blocks':self.num_blocks})
        return config

    def build(self, input_shape):
        encoder_features_shape = input_shape[0]
        self.num_encoder_vecs = encoder_features_shape[1]
        self.encoder_dim = encoder_features_shape[2]

        decoder_features_shape = input_shape[1]
        self.num_steps = decoder_features_shape[1]
        self.decoder_dim = decoder_features_shape[2]
        
        # mask
        ones = tf.ones([self.num_steps, self.num_steps])
        self.self_attention_mask = tf.linalg.band_part(ones, -1, 0)

    def call(self, inputs, training=False):
        encoder_features, decoder_features = inputs

        # add in positional encoding
        decoder_features = self.AddPositional(decoder_features)

        # loop through the attention and feed-forward blocks
        for self_att_layer, joint_att_layer, feed_layer in \
            zip(self.SelfAttentionBlocks, self.JointAttentionBlocks, self.FeedForwardBlocks):

            # masked self-attention
            decoder_features = self_att_layer([decoder_features, decoder_features, decoder_features],
                                               attention_mask=self.self_attention_mask, training=training)

            # joint attention (no mask)
            decoder_features = joint_att_layer([decoder_features, encoder_features, encoder_features],
                                                attention_mask=None, training=training)

            # feed forward
            decoder_features = feed_layer(decoder_features, training=training)

        return decoder_features

    def show_summary(self):    
        encoding = keras.layers.Input(shape=[self.num_encoder_vecs, self.encoder_dim], name='encoding')
        decoding = keras.layers.Input(shape=[self.num_steps, self.decoder_dim], name='decoding')
        inputs = [encoding, decoding]
        return tf.keras.Model(inputs, outputs=self.call(inputs), name=self.name).summary()

