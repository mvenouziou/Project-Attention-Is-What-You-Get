# imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

# custom imports
import tokenizers

"""
This file contains classes for preparing raw image and InChI data for use in our
model as Tensorflow Datasets. 
"""

class Pipeline:
    def __init__(self, parameters):
        self.parameters = parameters
        self.inchi_parser = InChiParsers(self.parameters)
        self.parsing_regex = self.inchi_parser.inchi_parsing_regex()
        self.tokenizers = tokenizers.Tokenizer(self.parameters)

    # Image loaders
    def load_image(self, image_path):
        image_path = tf.squeeze(image_path)
        image = keras.layers.Lambda(lambda x: tf.io.read_file(x))(image_path)
        return image   

    def decode_image(self, image, target_size):
        image = keras.layers.Lambda(lambda x: tf.io.decode_image(x, channels=3, expand_animations=False))(image)
        image = keras.layers.experimental.preprocessing.Resizing(*target_size)(image)
        return image    

    # extracts filepath from image name
    def path_from_image_id(self, x, root_folder):
        folder_a = tf.strings.substr(x, pos=0, len=1)
        folder_b = tf.strings.substr(x, pos=1, len=1)
        folder_c = tf.strings.substr(x, pos=2, len=1)
        filename =  tf.strings.join([x, '.png'])
        return tf.strings.join([root_folder, folder_a, folder_b, folder_c, filename], separator='/')

    def data_generator(self, image_set, labels_df=None, decode_images=True):
       
        parameters = self.parameters

        # get global params
        batch_size = parameters.batch_size()
        target_size = parameters.image_size()
        SOS = parameters.SOS()
        EOS = parameters.EOS()
        
        # dataset options
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = True
            
        # Train & Validation Datasets
        if image_set in ['train', 'valid']:
            root_folder = parameters.train_images_dir()  # train / valid images
            valid_split = 0.10
            
            # load labels into memory as dataframe
            if labels_df is None:
                labels_df = pd.read_csv(parameters.train_labels_csv())

            # test / train split
            num_valid_samples = int(valid_split * len(labels_df))
            train_df = labels_df.iloc[num_valid_samples: ]  # get train split
            valid_df = labels_df.iloc[: num_valid_samples]  # get validation split

            # shuffle
            train_df = train_df.sample(frac=1)
            valid_df = valid_df.sample(frac=1)

            # load into datasets  # (image_id, InChI)
            train_ds = tf.data.Dataset.from_tensor_slices(train_df.values)
            valid_ds = tf.data.Dataset.from_tensor_slices(valid_df.values)

            train_ds = train_ds.with_options(options)
            valid_ds = valid_ds.with_options(options)

            # update image paths  
            def map_path(x):  # (image_path, image_id, InChI)
                image_id = x[0]
                image_path = self.path_from_image_id(image_id, root_folder)
                return image_path, x[0], x[1]

            train_ds = train_ds.map(map_path, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(map_path, num_parallel_calls=tf.data.AUTOTUNE)

            def map_parse(x, y, z):  # (image_path, image_id, InChI)
                parsed_InChI = self.inchi_parser.parse_InChI_py_fn(z,self.parsing_regex)
                return x, y, parsed_InChI, z
    
            train_ds = train_ds.map(map_parse, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(map_parse, num_parallel_calls=tf.data.AUTOTUNE)
                    
            # load images into dataset       
            def open_images(w, x, y, z):
                w = self.load_image(w)
                return w, x, y, z
            
            train_ds = train_ds.map(open_images, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(open_images, num_parallel_calls=tf.data.AUTOTUNE)    

            # PREFETCH dataset BEFORE decoding images
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

            def decode(w, x, y, z):
                w = self.decode_image(w, target_size)
                return w, x, y, z

            if decode_images:
                train_ds = train_ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
                valid_ds = valid_ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)    

            # BATCH dataset AFTER decoding images (required by tf.io)
            # should batch before other pure TF Lambda layer ops
            train_ds = train_ds.batch(batch_size, drop_remainder=True)
            valid_ds = valid_ds.batch(batch_size, drop_remainder=True)
            
            # add extra "EOS" values to end of parsed inchi
            def extend_EOS(w, x, y, z):
                y = tf.strings.join([y, EOS, EOS, EOS, EOS, EOS], separator=' ')
                y = tf.reshape(y, [-1])
                return w, x, y, z

            train_ds = train_ds.map(extend_EOS, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(extend_EOS, num_parallel_calls=tf.data.AUTOTUNE)

            # Tokenize parsed_inchi.  Note: ds must be batched before this step (size=1 is ok) 
            train_ds = train_ds.map(self.tokenizers.tokenize_text, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(self.tokenizers.tokenize_text, num_parallel_calls=tf.data.AUTOTUNE)

            # name the elements
            def map_names(w, x, y, z):
                return  {'image': w, 'image_id': x, 'tokenized_InChI': y, 'InChI': z}
            
            train_ds = train_ds.map(map_names, num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(map_names, num_parallel_calls=tf.data.AUTOTUNE)
            
            return train_ds, valid_ds
        
        # Test Dataset
        elif image_set == 'test':

            # note: image resizing and batching done during this loading step
            # other elements must be batched before combining
            image_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory=parameters.test_images_dir(), labels='inferred', label_mode=None,
                class_names=None, color_mode='grayscale', batch_size=1, 
                image_size=target_size, shuffle=False, seed=None, validation_split=None, 
                subset=None, follow_links=False)

            # set filenames as label and batch
            image_id_ds = tf.data.Dataset.from_tensor_slices(image_ds.file_paths)
            image_id_ds = image_id_ds.map(lambda x: tf.strings.split(x, os.path.sep)[-1],
                                        num_parallel_calls=tf.data.AUTOTUNE)
            
            # prepare images for TF Records creations. 
            # Note: do this step AFTER filenames step
            if decode_images is False:  
                # convert image to raw byte string. Note: cannot have batch dim for encoding
                image_ds = image_ds.unbatch()
                image_ds = image_ds.map(lambda x: tf.cast(x, dtype=tf.uint16))
                image_ds = image_ds.map(lambda image: tf.io.encode_png(image))
                image_ds = image_ds.map(lambda image: tf.io.serialize_tensor(image))
                
            # dataset consisting solely of InChI start 'InChI=1S/'
            inchi_ds = image_id_ds.map(lambda x: tf.constant(SOS, dtype=tf.string),
                                    num_parallel_calls=tf.data.AUTOTUNE)
            
            # merge datasets
            test_ds = tf.data.Dataset.zip((image_ds, image_id_ds, inchi_ds, inchi_ds))
            
            # prefetch
            test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.batch(batch_size)

            # Tokenize parsed_inchi.  Note: ds must be batched before this step (size=1 is ok) 
            test_ds = test_ds.map(tokenize_text, num_parallel_calls=tf.data.AUTOTUNE)

            # set key names
            def map_names(w, x, y, z):
                return  {'image': w, 'image_id': x, 'tokenized_InChI': y, 'InChI': z}
            
            test_ds = test_ds.map(map_names, num_parallel_calls=tf.data.AUTOTUNE)
            
            return test_ds

    
class InChiParsers:
    def __init__(self, parameters):
        self.parameters = parameters

    def inchi_parsing_regex(self):
        # regex for spliting on InChi, but preserving chemical element abbreviations and three-digit numbers
        parameters = self.parameters

        # shortcut: hard coded values
        vocab = [parameters.EOS(), parameters.SOS(), '(',
                ')', '+', ',', '-', '/', 'Br', 'B', 'Cl', 'C', 'D', 'F',
                'H', 'I', 'N', 'O', 'P', 'Si', 'S', 'T', 'b', 'c', 'h', 'i',
                'm', 's', 't']
            
        vocab += [str(num) for num in reversed(range(168))]
        vocab = [re.escape(val) for val in vocab]
        
        """ # to create vocab from scratch, use:
        SOS = parameters.SOS()
        EOS = parameters.EOS()
        
        # load list of elements we should search for within InChI strings: 
        periodic_elements = pd.read_csv(PARAMETERS.periodic_table_csv(), header=None)[1].to_list()
        periodic_elements = periodic_elements + [val.lower() for val in periodic_elements] + [val.upper() for val in periodic_elements]
        
        punctuation = list(string.punctuation)
        punctuation = [re.escape(val) for val in punctuation]   # update values with regex escape chars added as needed

        three_dig_nums_list = [str(i) for i in range(1000, -1, -1)]

        vocab = [SOS, EOS] + periodic_elements + three_dig_nums_list + punctuation
        """

        split_elements_regex = rf"({'|'.join(vocab)})"
        
        return split_elements_regex

    def parse_InChI(self, texts, parsing_regex):  
        return ' '.join(re.findall(parsing_regex, texts))

    # TF dataset map-compatible version
    def parse_InChI_py_fn(self, texts, parsing_regex):

        def tf_parse_InChI(texts):  
            texts = np.char.array(texts.numpy())
            texts = np.char.decode(texts).tolist()
            texts = tf.constant([self.parse_InChI(val, parsing_regex) for val in texts])
            return tf.squeeze(texts)
        return tf.py_function(func=tf_parse_InChI, inp=[texts], Tout=tf.string)