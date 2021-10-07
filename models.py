import tensorflow as tf
from tensorflow import keras

# custom imports
import transformers
import prediction_heads
import tokenizers
import encoders

"""
This file integrates the components into our full model, together with 
inference's character-by-character text generation and fully parallelized training loops.

Architecture:
Image CNN Backbone --> Image Self-Attention Transformer --> 
    --> Image/Text Joint Attention Transformer --> Prediction

For inference, inputs should be a list with batch of images at the first index.
For training / evaluation the tokenized InChi value must be included at the second index
"""

class MolecularTranslator(keras.Model):

    def __init__(self, encoder_blocks, encoder_dim, decoder_blocks, decoder_dim, 
                 parameters, dual_heads_split_step=None, use_convolutions=False,  
                 name='MolecularTranslator', **kwargs):

        super().__init__(name=name, **kwargs)

        self.encoder_blocks = encoder_blocks
        self.encoder_dim = encoder_dim
        self.decoder_blocks = decoder_blocks
        self.decoder_dim = decoder_dim
        self.parameters = parameters
        self.dual_heads_split_step = dual_heads_split_step
        self.use_convolutions = use_convolutions

        # hard coded params
        self.num_encoder_heads = 8
        self.num_decoder_heads = 8
        self.backbone_image_shape = [320, 320]  # all images will be rescaled to this before processing

    def get_config(self):
        config = {'encoder_blocks': self.encoder_blocks,
                  'encoder_dim': self.encoder_dim,
                  'decoder_blocks':self.decoder_blocks,
                  'decoder_dim': self.decoder_dim,
                  'dual_heads_split_step': self.dual_heads_split_step,
                  'use_convolutions': self.use_convolutions,
                  'parameters':self.parameters,
        }
        return config 


    def build(self, input_shape):

        # construct tokenizer
        tokenizer_obj = tokenizers.Tokenizer(parameters=self.parameters)

        self.tokenizer_layer = tokenizer_obj.tokenizer_layer()
        self.inverse_tokenizer = tokenizer_obj.inverse_tokenizer()
        self.tokenized_EOS = tokenizer_obj.tokenized_EOS()  
        self.vocab_size = self.tokenizer_layer.vocabulary_size()
        self.EOS = self.parameters.EOS()
       
        # InChI encoding
        self.InChIEncoder = encoders.InChIEncoder(vocab_size=self.vocab_size, 
                                         embedding_dim=self.decoder_dim, 
                                         use_convolutions=self.use_convolutions)

        # Image CNN
        self.ImageEncoderBackbone = encoders.ImageEncoderBackbone(self.backbone_image_shape)
        self.ImageDownscaler = encoders.ImageDownscaler(encoder_dim=self.encoder_dim)
        
        # Transformers
        self.EncoderAttention = transformers.EncoderAttention(num_attention_heads=self.num_encoder_heads, 
                                                              num_blocks=self.encoder_blocks)

        self.DecoderAttention = transformers.DecoderAttention(num_blocks=self.decoder_blocks, 
                                                 num_attention_heads=self.num_decoder_heads)
        
        # Predictions 
        self.DecoderHead = prediction_heads.DecoderHead(vocab_size=self.vocab_size, 
                                       dual_heads_split_step=self.dual_heads_split_step)
        
    @tf.function
    def call(self, inputs, training=False):

        image = inputs[0]
        tokenized_inchi = inputs[1]

        # encoder
        encoder_features = self.ImageEncoderBackbone(image, training=training)
        encoder_features = self.ImageDownscaler(encoder_features, training=training)           
        encoder_features = self.EncoderAttention(encoder_features, training=training)

        # decoder
        if not training:
            # character-by-character generation loop
            probabilities, _ = self.generation_loop([encoder_features, tokenized_inchi], training=False)

        else:  
            # parallel computations with masked attention
            decoder_features = self.InChIEncoder(tokenized_inchi, training=training)

            decoder_features = self.DecoderAttention(
                [encoder_features, decoder_features], training=training)

            # predictions
            probabilities = self.DecoderHead(decoder_features, training=training)

        return probabilities

    # inference yielding generated probabilities from a single batch input
    def generation_loop(self, inputs, training=False):
        
        encoder_features = inputs[0]
        tokenized_inchi = inputs[1]

        # get shapes
        batch_size = tf.shape(tokenized_inchi)[0]
        padded_length = tf.shape(tokenized_inchi)[1]
               
        # decoder
        # create containers
        generated_probs = tf.TensorArray(dtype=tf.float32, size=padded_length,
                element_shape=tf.TensorShape([None, self.vocab_size]))

        generated_inchi = tf.TensorArray(dtype=tf.int32, size=padded_length,
                element_shape=tf.TensorShape([None]))
        
        # initialize generated probs array
        zeros = tf.zeros((batch_size, self.vocab_size), dtype=generated_probs.dtype)
        for i in range(padded_length):
            generated_probs = generated_probs.write(i, zeros)

        # initialize generated InChI values array
        zeros = tf.zeros((batch_size), dtype=generated_inchi.dtype)
        for i in range(padded_length):
            generated_inchi = generated_inchi.write(i, zeros)

        # initialize step
        step = tf.constant(0, dtype=tf.int32)
        
        # loop body function
        def body_fn(generated_inchi, generated_probs, step):

            inchi = tf.transpose(generated_inchi.stack(), [1,0])
            
            # get current step probs and save result
            probs = self.decoder_step(encoder_features, tokenized_inchi, step)
            generated_probs = generated_probs.write(step, tf.cast(probs, dtype=generated_probs.dtype))

            # get new token prediction and save result
            predicted_token = tf.argmax(probs, axis=-1)
            generated_inchi = generated_inchi.write(
                    step, tf.cast(predicted_token, dtype=generated_inchi.dtype))

            # update step
            step = step + 1
            step = tf.cast(step, dtype=step.dtype)
            
            return [generated_inchi, generated_probs, step]

        # loop conditional function
        def cond_fn(generated_inchi, generated_probs, step):
            return tf.math.less(step, padded_length)

        # run generation loop
        generated_inchi, generated_probs, step = \
            tf.while_loop(cond=cond_fn,
                          body=body_fn,
                          loop_vars=[generated_inchi, generated_probs, step],
                          parallel_iterations=1,
                          maximum_iterations=padded_length,
                          shape_invariants=[None, None, tf.TensorShape([])],
                          )
        
        # unpack generated probabilities
        probabilities = tf.transpose(generated_probs.stack(), [1, 0, 2])
        inchi = tf.transpose(generated_inchi.stack(), [1, 0])

        return probabilities, inchi
    
    def decoder_step(self, encoder_features, tokenized_inchi, step):

        # decoder
        decoder_features = self.InChIEncoder(tokenized_inchi, training=False)
        decoder_features = self.DecoderAttention([encoder_features, decoder_features], training=False)
        
        # get probabilities
        probabilities = self.DecoderHead(decoder_features, training=False)[:, step, :]

        return probabilities

    def tokens_to_string(self, token_predictions):

        # convert to strings
        parsed_string_vals = self.inverse_tokenizer(token_predictions)
        string_vals = keras.layers.Lambda(
            lambda x: tf.strings.reduce_join(x, axis=-1))(parsed_string_vals)

        # remove first EOS generated and everything after
        pattern = ''.join([self.EOS, '.*$'])
        string_vals = tf.strings.regex_replace(string_vals, pattern, rewrite='', 
                                               replace_global=True, name='remove_EOS')   

        return string_vals