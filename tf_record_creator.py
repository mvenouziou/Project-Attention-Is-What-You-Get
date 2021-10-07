import tensorflow as tf
import pipeline
import os

"""
This class is used to create and decode TF Record Shards required for optimal TPU training. 
Note that TPU requires the shards to be stored with Google Cloud Storage buckets.
"""


class TFRecordCreator(pipeline.Pipeline):
    def __init__(self, parameters):
        self.parameters = parameters

    # Create TF Examples
    def make_example(self, image, image_id, tokenized_InChI, InChI):
        image_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image.numpy()])  # image provided as raw bytestring
        )
        image_id_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_id.numpy()])
        )
        tokenized_InChI_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokenized_InChI).numpy()])
        )
        InChI_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[InChI.numpy()])
        )

        features = tf.train.Features(feature={
            'image': image_feature,
            'image_id': image_id_feature,
            'tokenized_InChI': tokenized_InChI_feature,
            'InChI': InChI_feature
        })
        
        example = tf.train.Example(features=features)

        return example.SerializeToString()


    def make_example_py_fn(self, image, image_id, InChI, tokenized_InChI):
        return tf.py_function(func=self.make_example, 
                    inp=[image, image_id, InChI, tokenized_InChI], 
                    Tout=tf.string)


    # Decode TF Examples
    def decode_example(self, example):        

        parameters = self.parameters
        
        feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
                            'image_id': tf.io.FixedLenFeature([], tf.string),
                            'tokenized_InChI': tf.io.FixedLenFeature([], tf.string),
                            'InChI': tf.io.FixedLenFeature([], tf.string)}
        
        values = tf.io.parse_single_example(example, feature_description)
        
        
        values['image'] = self.decode_image(values['image'], parameters.image_size())
        values['tokenized_InChI'] = tf.io.parse_tensor(values['tokenized_InChI'],
                                                    out_type=tf.int64)
        values['tokenized_InChI'] = tf.cast(values['tokenized_InChI'], tf.int32)
        
        return values


    def serialized_dataset_gen(self, set_type='train', labels_df=None):
        
        parameters = self.parameters

        if set_type == 'train':
            train_ds, valid_ds = data_generator(image_set='train', 
                                                parameters=parameters, 
                                                labels_df=train_labels_df, 
                                                decode_images=False)  # output images as bytestrings

            train_ds = train_ds.unbatch()
            valid_ds = valid_ds.unbatch()

            # Create TF Examples
            train_ds = train_ds.map(lambda x: self.make_example_py_fn(x['image'], x['image_id'], x['tokenized_InChI'], x['InChI']), 
                                    num_parallel_calls=tf.data.AUTOTUNE)
            valid_ds = valid_ds.map(lambda x: self.make_example_py_fn(x['image'], x['image_id'], x['tokenized_InChI'], x['InChI']), 
                                    num_parallel_calls=tf.data.AUTOTUNE)
            
            return train_ds, valid_ds
        
        else: #test_set:
            test_ds = data_generator(image_set='test', 
                                    parameters=parameters, 
                                    labels_df=None, 
                                    decode_images=False)  # output images as bytestrings
            
            test_ds = test_ds.unbatch()
                
            # Create TF Examples
            test_ds = test_ds.map(lambda x: self.make_example_py_fn(x['image'], x['image_id'], x['tokenized_InChI'], x['InChI']), 
                                num_parallel_calls=tf.data.AUTOTUNE)
            
            return test_ds

    # Create TF Record Shards
    """
    NOTE: Changes have been made to the other dataset pipeline functions. 
    Test / Revise this for compatability before running.
    """
    def create_records(self, dataset, subset, num_shards):
        
        folder = subset + '_tfrec'
        
        if subset =='train':
            num_samples = int(.9 * len(train_labels_df))    # test / valid split
        elif subset == 'valid':
            num_samples = int(.1 * len(train_labels_df))
        else:
            num_samples = 2000000

        if not os.path.isdir(folder):
            os.mkdir(folder)
            
        for shard_num in range(num_shards):
            
            filename = os.path.join(folder, f'{subset}_shard_{shard_num+1}')
            try:
                this_shard = dataset.skip(shard_num * num_samples//num_shards).take(num_samples//num_shards)
            
                print(f'Writing shard {shard_num+1}/{num_shards} to {filename}')
                writer = tf.data.experimental.TFRecordWriter(filename)
                writer.write(this_shard)
            except:
                break
        return None 
    

    # Load dataset from saved TF Record Shards
    def dataset_from_records(self, subset):

        parameters = self.parameters

        # optimizations
        options = tf.data.Options()
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.apply_default_optimizations = True

        filepath = os.path.join(parameters.tfrec_dir(), 
                                subset + '_tfrec/*')

        dataset = tf.data.Dataset.list_files(filepath)  # put all tf rec filenames in a ds
        dataset = dataset.shuffle(10**6)
    
        # merge the files
        num_readers = parameters.strategy().num_replicas_in_sync
        dataset = dataset.interleave(tf.data.TFRecordDataset,  
                                    cycle_length=num_readers, block_length=1,
                                    deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.shuffle(10**6)
        
        # decode examples
        dataset = dataset.map(self.decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # note: tokenized InChI element spec needs help determining shape
        for val in dataset.take(1):
            padded_length = val['tokenized_InChI'].shape[-1]

        # coerce unknown shape
        dataset = dataset.map(lambda x: {'image':x['image'],
                                        'image_id': x['image_id'],
                                        'tokenized_InChI': tf.reshape(x['tokenized_InChI'], [padded_length]),
                                        'InChI': x['InChI']},
                            num_parallel_calls=tf.data.AUTOTUNE)  

        dataset = dataset.batch(parameters.batch_size(), drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
        return dataset