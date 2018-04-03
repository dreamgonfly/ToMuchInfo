import tensorflow as tf
import numpy as np

class BaselineRegression:
     
    def __init__(self, dictionary, config):
        
        self.vocabulary_size = dictionary.vocabulary_size
        self.embedding_size = dictionary.embedding_size - 1 # except for padding
        self.sentence_length = config.sentence_length
        
        self.kernel_num = 100
        self.kernel_sizes = [3,4,5]

        self.reviews = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='reviews')
        self.features = tf.placeholder(tf.int32, shape=[None, 1], name='features')
        self.labels = tf.placeholder(tf.float32, shape=[None], name='labels') # will be used to get loss and accuracy
    
    def build_model(self, scope=None, reuse=None):
        
        with tf.variable_scope(scope, default_name='D', reuse=reuse):
            self.Wemb = tf.get_variable('Wemb', shape=[self.vocabulary_size, self.embedding_size])
            embedded = tf.nn.embedding_lookup(self.Wemb, self.reviews)
#             print(embedded.eval())
#             embedded = tf.contrib.layers.fully_connected(x_hat, self.embed_size, activation_fn=None)
            self.pools = []
            for kernel_size in self.kernel_sizes:
                kernel = tf.get_variable('kernel_'+str(kernel_size), shape=[kernel_size, self.embedding_size, self.kernel_num])
                self.feature_map = tf.nn.conv1d(embedded, kernel, stride=1, padding='SAME')
                expanded = tf.expand_dims(self.feature_map, axis=1)
                pooled = tf.nn.max_pool(expanded, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
                self.pools.append(tf.squeeze(pooled, axis=1))
                
            self.conv_concat = tf.contrib.slim.flatten(tf.concat(self.pools, axis=2))
            self.logits = tf.contrib.layers.fully_connected(self.conv_concat, 1, activation_fn=None)
            self.predictions = tf.squeeze(self.logits, axis=1)
            return self.predictions