import random
import json
import tensorflow as tf
from tensorflow.keras.utils import plot_model

import model
from dataset import *

DATA_FILE = "cleaned_shuffled.json"
CHECK = "checkpoint/cp-{epoch:04d}.ckpt" 

d = open(DATA_FILE)
raw_data = json.loads(d.read())


anime_file = open("cleaned_shuffled.json")
anime_data = anime_file.read()

ds = AnimeDataset(anime_data, "../small_result").batch(50).prefetch(5)
val_ds = AnimeDataset(anime_data, "../small_result", training=False).batch(50).prefetch(5)

corpus = generate_corpus(anime_data)

classifier, anime_encoder = model.build_model(corpus)
plot_model(classifier)

losses = {}
loss = tf.keras.losses.BinaryCrossentropy()
for genre in GENRES_NORMALIZED:
    losses[genre] = loss

classifier.compile(optimizer='adam', 
            loss=losses,
            metrics=['accuracy'])

#classifier.load_weights("checkpoint/cp-0010.ckpt")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=CHECK, save_weights_only=True, verbose=2) 
]


classifier.fit(ds, verbose=2, epochs=10, callbacks=callbacks)
anime_encoder.desc_encoder.dump_word_embeddings()
