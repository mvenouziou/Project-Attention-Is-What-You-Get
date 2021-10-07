import tensorflow as tf
import pandas as pd

"""
The Tokenizer class is used to tokenize InChi values. As vocabulary it uses the
periodic table elements (appearing in the training set) along with some additional 
characters found in the training set and SOS / EOS values.

An inverse tokenizer is also included for deciphering model predictions.
"""
class Tokenizer:
    """ tokenizes, crops & pads to parameters.padded_length() """

    def __init__(self, parameters):
        self.parameters = parameters

        vocab = self.create_vocab()
        
        # create tokenizer
        tokenizer_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=None, split=lambda x: tf.strings.split(x, sep=' ', maxsplit=-1), 
            output_mode='int', output_sequence_length=self.parameters.padded_length(), 
            vocabulary=vocab)

        # record EOS token
        tokenized_EOS = tokenizer_layer(tf.constant([self.parameters.EOS()]))
        
        # create inverse (de-tokenizer)
        inverse_tokenizer = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=vocab, invert=True)

        self._tokenizer_layer = tokenizer_layer
        self._inverse_tokenizer = inverse_tokenizer
        self._tokenized_EOS = tokenized_EOS
    
    # Create vocabulary for tokenizer
    def create_vocab(self):       

        PARAMETERS = self.parameters

        hard_coded_vocab = [PARAMETERS.EOS(), PARAMETERS.SOS(), '(',
            ')', '+', ',', '-', '/', 'B', 'Br',  'C', 'Cl', 'D', 'F',
            'H', 'I', 'N', 'O', 'P', 'S', 'Si', 'T', 'b', 'c', 'h', 'i',
            'm', 's', 't']
        
        numbers = [str(num) for num in range(168)]
        
        vocab = hard_coded_vocab + numbers
        
        """
        # get from saved file
        vocab = pd.read_csv(PARAMETERS.vocab_csv())['vocab_value'].to_list()   
        vocab = list(vocab)
        """

        """ 
        # To create from scratch, extract all vocab elements appearing in train set:
        df = pd.read_csv(PARAMETERS.train_labels_csv())  
        seg_len = 250000
        num_breaks = len(df) // seg_len

        vocab = set()
        for i in range(num_breaks):

            df_i =  df['InChI'].iloc[seg_len * i: seg_len * (i+1)]
            texts =  df_i.apply(lambda x: set(parse_InChI(x).split()))
            texts = texts.tolist()

            vocab = vocab.union(*texts)

            print(f'completed {i} / {num_breaks}')

        vocab = list(vocab)
        vocab_df = pd.DataFrame({'vocab_value': vocab})

        # save results
        filename = os.path.join(PARAMETERS.csv_save_dir(), 'vocab.csv')
        vocab_df.to_csv(filename, index=False)
        """
            
        return vocab

    def tokenizer_layer(self):
        return self._tokenizer_layer
    
    def inverse_tokenizer(self):
        return self._inverse_tokenizer
        
    def tokenized_EOS(self):
        return self._tokenized_EOS

    def tokenize_text(self, w, x, y, z):
        # note: requires batch dim
        y = self._tokenizer_layer(y)
        return w, x, y, z