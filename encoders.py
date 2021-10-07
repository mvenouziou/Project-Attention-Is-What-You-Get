import tensorflow as tf
from tensorflow import keras

"""
Initial encoding steps for Images and (tokenized) text, before going through the
attention transformers. 

Images are run through a CNN "backbone" network from the Keras library, with 
a few additional convolutional layers added on top to adjust the feature dimension.

If training on TPU this CNN is trained from scratch. Otherwise, pretrained weights 
should be loaded and locked until the transformers start to yield reasonable results.

InChi arrive tokenized. Here they are split into offset input/target pairs, 
passed through an embedding, and given a trainable start token. Optionally, 
the tokens can also be run through a few masked convolutional layers.
"""


class PrependStartVar(tf.keras.layers.Layer):

    def __init__(self, name='PrependStartVar', **kwargs):
        super().__init__(name=name, **kwargs)

        self.Concat = keras.layers.Concatenate(axis=1)
        
    def config(self):
        return super().get_config()

    def build(self, input_shape):
        InChI_embedding_shape = input_shape
        embedding_dim = InChI_embedding_shape[-1]

        # trainable start variable
        initializer = tf.random_normal_initializer()(shape=[1, embedding_dim])
        self.start_var = tf.Variable(initializer, trainable=True, name='start_var')

    def call(self, inputs):
        InChI_embedding = inputs
        batch_size = tf.shape(InChI_embedding)[0]
        

        InChI_embedding = self.Concat([tf.tile(tf.expand_dims(self.start_var, axis=0),
                                                [batch_size, 1, 1]),
                                        InChI_embedding])

        return InChI_embedding


class InChIEncoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_dim, use_convolutions=False, name='InChIEncoder', **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_convolutions = use_convolutions

    def config(self):
        config = super().get_config()
        config.update({'vocab_size': self.vocab_size, 'embedding_dim': self.embedding_dim, 
                       'use_convolutions': self.use_convolutions})
        return config

    def build(self, input_shape):

        inchi_shape = input_shape
        self.num_chars = inchi_shape[1]


        # layers
        self.PrependStartVar = PrependStartVar(name='PrependStartVar')
        self.EmbeddingLayer = tf.keras.layers.Embedding(input_dim=self.vocab_size, 
            output_dim=self.embedding_dim, mask_zero=False, input_length=self.num_chars-1)

        self.Add = keras.layers.Add()
        self.Concat = keras.layers.Concatenate(axis=1)
        self.DepthwiseConv2D = keras.layers.DepthwiseConv2D(kernel_size=3, 
                        strides=1, padding='same', data_format='channels_first',
                        activation='relu')
        self.Dense = keras.layers.Dense(1, activation='relu')
          
    def call(self, inputs):
        
        inchi = inputs

        # embedding
        inchi = inchi[:, :-1]  # drop last val
        inchi = self.EmbeddingLayer(inchi)
        
        # (Optional: masked convolution)
        if self.use_convolutions:
            
            # extend to (batch, len, len, dim) and mask for parallelized convolutions
            ones = tf.ones((self.num_chars-1, self.num_chars-1))
            mask = tf.linalg.band_part(ones, -1, 0)
            mask = tf.reshape(mask, [1, self.num_chars-1, 1, self.num_chars-1])

            inchi = tf.tile(tf.expand_dims(inchi, -1), [1, 1, 1, self.num_chars-1])
            inchi = inchi * tf.cast(mask, dtype=inchi.dtype)

            # apply parallel convolutions (maintains independence to mask future steps)
            inchi = self.DepthwiseConv2D(inchi)

            # squeeze out last dim
            inchi = self.Dense(inchi)
            inchi = tf.squeeze(inchi, axis=-1)

        # append start token
        inchi = self.PrependStartVar(inchi)
        
        return inchi

    def show_summary(self):    
        inchi = keras.layers.Input([self.num_chars], name='tokenized_inchi')
        inputs = inchi
        return tf.keras.Model(inputs, outputs=self.call(inputs), name=self.name).summary()


class ImageEncoderBackbone(keras.layers.Layer):
    """ Note: efficientnet requires input images to have unscaled float values (in [0, 255]) """
    
    def __init__(self, image_shape, name='ImageEncoderBackbone', **kwargs):
        # Note: disable mixed precision for this layer (Transfer model doesn't handle it properly)
        super().__init__(name=name, dtype=tf.float32, **kwargs)
        
        self.image_shape = image_shape
        height = self.image_shape[0]
        width = self.image_shape[1]

        base_transfer_model = keras.applications.EfficientNetB3(
                                include_top=False, 
                                weights=None,
                                input_shape=[height, width, 3])

        self.preprocessor = tf.keras.applications.efficientnet.preprocess_input
        self.Resize = keras.layers.Resizing(height=height, width=width, name='Resizing')
        
        self.transfer_model = keras.Model(inputs=base_transfer_model.inputs, 
                                    outputs=base_transfer_model.get_layer('top_activation').output, 
                                    name='EfficientNet')

    
    def config(self):
        config = super().get_config()
        config.update({'image_shape': self.image_shape})
        return config
    
    def call(self, inputs):
        image = inputs        
        image = self.Resize(image)
        image = tf.cast(image, dtype=tf.float32)
        image = self.preprocessor(image)
        image = self.transfer_model(image)
        return image

    def show_summary(self):
        image = tf.keras.layers.Input(shape=[448, 448, 3], name='image')
        inputs = image
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()


class ImageDownscaler(keras.layers.Layer):

    def __init__(self, encoder_dim, name='ImageDownscaler', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.encoder_dim = encoder_dim

        # layers
        self.Conv2D_1 = keras.layers.Conv2D(filters=self.encoder_dim, kernel_size=1, 
                                            activation='relu', name='Conv2D_1')
        self.Conv2D_2 = keras.layers.Conv2D(filters=self.encoder_dim, kernel_size=1, 
                                            activation='relu', name='Conv2D_2')
        self.Add = keras.layers.Add(name='add_positional_encoding')
    
    def config(self):
        config = super().get_config()
        config.update({'encoder_dim':self.encoder_dim})
        return config

    def build(self, input_shapes):
        self.encoder_features_shape = input_shapes

    def call(self, inputs):
        encoder_features = inputs
        
        # update dims
        encoder_features = self.Conv2D_1(encoder_features)
        encoder_features = self.Conv2D_2(encoder_features)

        return encoder_features

    def show_summary(self):
        image = tf.keras.layers.Input(shape=self.encoder_features_shape[1:], name='image')
        inputs = image
        outputs = self.call(inputs)  # note: use self.call() for all layers to show in summary
        return tf.keras.Model(inputs, outputs, name=self.name).summary()



