from zipfile import ZipFile
import tensorflow as tf
import os

class ModelParameters:
    def __init__(self, cloud_server='kaggle'):

        # check for TPU & initialize
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
            tf.config.experimental_connect_to_cluster(resolver)

            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All devices: ", tf.config.list_logical_devices('TPU'))

            STRATEGY = tf.distribute.TPUStrategy(resolver)
            TPU = True
            os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"  # for TF Hub models on TPU
            PRECISION_TYPE = 'mixed_bfloat16' 

        except:
            TPU = False
            if tf.config.list_physical_devices('GPU'):  # check for GPU
                PRECISION_TYPE = 'mixed_float16'
            else:
                PRECISION_TYPE = 'float32'
            
            STRATEGY = tf.distribute.get_strategy()

        # enable mixed precision
        tf.keras.mixed_precision.set_global_policy(PRECISION_TYPE)
               
        # universal parameters
        self._batch_size = 16  # used on GPU. TPU batch size increased below
        self._padded_length = 200
        self._image_size = (320, 320)  # shape to process images in data pipeline. Size is restricted by memory constraints.
        self.SOS_string = 'InChI=1S/'  # start of sentence value
        self.EOS_string = '<EOS>'  # end of sentence value
        self._strategy = STRATEGY
        self._precision_type = PRECISION_TYPE
        self._tpu = TPU
        
        # TPU batch size
        if self._tpu:
            # note: utilize steps_per_execution compile parameter to increase TPU throughput
            self._batch_size = 64 * self._strategy.num_replicas_in_sync  

        # File Paths       
        if cloud_server == 'colab':  # Google Colab

            archive_dir = os.path.join('drive', 'MyDrive/datasets/BristolMeyer')
            local_dir = os.path.join('bms-molecular-translation', '')
            checkpoint_dir =  os.path.join(archive_dir, 'checkpoints')
            
            # check for TPU 
            if self._tpu: 

                # TPU file structure (via Kaggle GCS folder)
                
                self._dataset_dir = 'gs://kds-391ec311f31a9ea8257ee03d877d162389c216bf2612a6c0240375a8' # Get updated directory on Kaggle via KaggleDatasets().get_gcs_path('bms-molecular-translation')
                self._prepared_files_dir = 'gs://kds-65441605ef46434facd4b7a78d5eab7c7e0d48898e6942b4d0dcaea6'  # Get updated directory on Kaggle via KaggleDatasets().get_gcs_path('periodic-table')
                self._tfrec_dir = 'gs://kds-391ec311f31a9ea8257ee03d877d162389c216bf2612a6c0240375a8'  # from Kaggle. Get updated directory on Kaggle via KaggleDatasets().get_gcs_path('bmsshards')
                self._checkpoint_dir =  checkpoint_dir
                self._load_checkpoint_dir = self._checkpoint_dir
                self._csv_save_dir = './'

            else:
                # unzip data
                if not os.path.isdir(local_dir):
                    source_file = os.path.join(archive_dir, 'bms-molecular-translation.zip')

                    with ZipFile(source_file, 'r') as zip:
                        zip.extractall(local_dir)

                # file paths
                self._dataset_dir = local_dir
                self._prepared_files_dir = archive_dir
                self._checkpoint_dir = checkpoint_dir
                self._load_checkpoint_dir = self._checkpoint_dir
                self._csv_save_dir = self._prepared_files_dir 
                self._tfrec_dir = None
                
        elif cloud_server == 'kaggle': # Kaggle cloud notebook (CPU / GPU)
            from kaggle_datasets import KaggleDatasets
            
            # check for TPU 
            if self._tpu: 
                
                # file paths
                self._dataset_dir = '' #KaggleDatasets().get_gcs_path('bms-molecular-translation')
                self._prepared_files_dir = KaggleDatasets().get_gcs_path('periodic-table')
                self._tfrec_dir = KaggleDatasets().get_gcs_path('bmsshards')
                self._checkpoint_dir = './'
                self._load_checkpoint_dir = './'
                self._csv_save_dir = './'

            # set GPU instance info
            else:  
                # file paths
                self._dataset_dir = '../input/bms-molecular-translation/'
                self._prepared_files_dir = '../input/periodic-table/'
                self._tfrec_dir = '../input/bmsshards/'
                self._checkpoint_dir = './'
                self._load_checkpoint_dir = '../input/k/mvenou/bms-molecular-translation/checkpoints/'
                self._csv_save_dir = './'
                self._tfrec_dir = None

        # common file paths
        self._periodic_table_csv = os.path.join(self._prepared_files_dir, 'periodic_table_elements.csv')
        self._vocab_csv = os.path.join(self._prepared_files_dir, 'vocab.csv')        
        self._test_images_dir = os.path.join(self._dataset_dir, 'test/')
        self._train_images_dir = os.path.join(self._dataset_dir, 'train/')
        self._extra_labels_csv = os.path.join(self._dataset_dir, 'extra_approved_InChIs.csv')
        self._train_labels_csv = os.path.join(self._dataset_dir, 'train_labels.csv')
        self._sample_submission_csv = os.path.join(self._dataset_dir, 'sample_submission.csv')
        
    # functions to access params
    def padded_length(self):
        return self._padded_length
    def mixed_precision(self):
        return self._precision_type
    def tpu(self):
        return self._tpu
    def tfrec_dir(self):
        return self._tfrec_dir
    def cloud_server(self):
        return self._cloud_server
    def strategy(self):
        return self._strategy
    def csv_save_dir(self):
        return self._csv_save_dir
    def train_labels_csv(self):
        return self._train_labels_csv
    def vocab_csv(self):
        return self._vocab_csv
    def periodic_table_csv(self):
        return self._periodic_table_csv
    def batch_size(self):
        return self._batch_size  
    def image_size(self):
        return self._image_size    
    def SOS(self):
        return self.SOS_string
    def EOS(self):
        return self.EOS_string
    def train_images_dir(self):
        return self._train_images_dir
    def test_images_dir(self):
        return self._test_images_dir   
    def checkpoint_dir(self):
        return self._checkpoint_dir
    def load_checkpoint_dir(self):
        return self._load_checkpoint_dir
