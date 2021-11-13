import numpy as np
import random
import json
import tensorflow as tf
import model
import matplotlib.pyplot as plt

from dataset import *

DATA_FILE = "cleaned_shuffled.json"

anime_file = open(DATA_FILE)
anime_data = anime_file.read()

ds = AnimeDataset(anime_data, "../small_result", training=False).batch(5)

corpus = generate_corpus(anime_data)

classifier, anime_encoder = model.build_model(corpus=corpus)
classifier.compile(optimizer='adam', 
            metrics=['accuracy'])

classifier.load_weights("checkpoint/cp-0010.ckpt")

for data in ds.as_numpy_iterator():
    predictions = classifier.predict(data[0], batch_size=5)
    corrected = np.array(predictions).transpose()

    plt.figure(figsize=(10, 15))

    for i in range(len(data[0]["img"])):
        plt.subplot(5, 2, i*2 + 1)
        plt .grid(False)
        plt.yticks([0, 0.5, 1])
        plt.xticks(rotation=75)
        bar = plt.bar(GENRES, corrected[0][i], color='#000077')
        plt.ylim([0,1])

        plt.subplot(5, 2, i*2 + 2)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data[0]["img"][i] + 0.5)

        desc = data[0]["desc"][i]
        desc = desc[0:100] if len(desc) > 100 else desc

        plt.xlabel(desc, wrap=True)

        for gInd, gId in enumerate(GENRES_NORMALIZED):
            if data[1][gId][i]:
                bar[gInd].set_color("green")

    plt.tight_layout()
    plt.show()
