import io
import string

from os.path import exists
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

import dataset

VOCAB_SIZE = 5000
EMBEDDING_DIM = 128 


#FIXME: We can just use Sequential instead.

class CoverImageLayer(Layer):
    """
    The part of the model that processes the cover image of the anime.
    The returned value is the information we extracted from the cover image.
    """

    def __init__(self, name="CoverImageLayer", **kwargs):
        super(CoverImageLayer, self).__init__(name=name, **kwargs)


        # Use data augmentation to get better generalization.
        self.data_augmentation = tf.keras.Sequential([
            RandomFlip("horizontal",input_shape=(150,100,3)),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ])

        self.stack = tf.keras.Sequential([
            self.data_augmentation,
            Conv2D(16, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),

            Conv2D(128, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            BatchNormalization(),

            Flatten(),

            Dense(256, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),

            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.5),
            BatchNormalization(),
        ])

    def call(self, inputs):
        return self.stack(inputs)

#TODO:Try fine tuning a bert instead
class DescLayer(Layer):
    """
    The part of the model that processes the anime description.
    """
    def normalize_text(self, text_input):
        """
        Remove HTML tags and punctuation
        """
        text_input = tf.strings.lower(text_input)
        text_input = tf.strings.regex_replace(text_input, '<[^>]*>', '')
        text_input = tf.strings.regex_replace(text_input, "[%s]" % string.punctuation, '')
        return text_input

    def __init__(self, name="DescLayer", corpus=None, word_dict=None, **kwargs):
        """
            corpus can be used for extracting a dictionary if provided.
        """
        super(DescLayer, self).__init__(name=name, **kwargs)

        self.text_vectorize = TextVectorization(max_tokens=VOCAB_SIZE, 
            standardize=self.normalize_text,
            output_mode='int',
            name='vectorize_layer')
        
        if corpus:
            self.text_vectorize.adapt(corpus)
        elif word_dict:
            self.text_vectorize.set_dict(word_dict)        

        self.embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM, name='embedding')

        self.stack = tf.keras.Sequential([ 
            self.text_vectorize,
            self.embedding,

            Bidirectional(GRU(256, return_sequences=True)),

            Bidirectional(GRU(128, return_sequences=True)),

            Bidirectional(GRU(64)),

            Dense(64, activation='relu'),
            Dropout(0.3),
            BatchNormalization(),

            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5)
        ])
    
    def call(self, inputs):
        return self.stack(inputs)
    
    def dump_word_embeddings(self):
        """
        Dump the learned word embeddings into a file.
        """
        weights = self.embedding.get_weights()[0]
        vocab = self.text_vectorize.get_vocabulary()

        out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
        out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

        for index, word in enumerate(vocab):
            if index == 0:
                continue  # skip 0, it's padding.
            vec = weights[index]
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")
            out_m.write(word + "\n")
        out_v.close()
        out_m.close()



class AnimeEncoder(Layer):
    """
        This Layer combines CoverImaegLayer and DescLayer.
    """
    def __init__(self, corpus=None, word_dict=None, **kwargs):
        super(AnimeEncoder, self).__init__(**kwargs)
        
        self.cover_image_encoder = CoverImageLayer()
        self.desc_encoder = DescLayer(corpus=corpus, word_dict=word_dict)

        self.combine = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu')
        ])


    def call(self, inputs):
        cover_info = self.cover_image_encoder(inputs["img"])
        desc_info = self.desc_encoder(inputs["desc"])
        
        return self.combine(tf.concat([cover_info, desc_info], axis=1))


def build_model(corpus):
    model_input = {"img": Input(shape=(150, 100, 3), dtype="float32", name='img'),
         "desc": Input(shape=(1), dtype="string", name='desc')}

    anime_encoder = AnimeEncoder(corpus=corpus)
    x = anime_encoder(model_input)

    output_values = []
    # Build the individual classifiers.
    for genre in dataset.GENRES_NORMALIZED:
        y = Dropout(0.5)(x)
        y = Dense(64, activation='relu')(y)
        y = Dense(1, activation='sigmoid', name=genre)(y) 
        output_values.append(y)
    
    return tf.keras.Model(inputs=model_input, outputs=output_values), anime_encoder
