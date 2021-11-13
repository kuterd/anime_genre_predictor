import json
import tensorflow as tf
from os.path import exists


GENRES = ['Romance', 'Slice of Life', 'Music', 'Thriller', 'Comedy', 'Supernatural', 'Adventure', 'Mecha', 'Drama', 'Ecchi', 'Psychological', 'Fantasy', 'Hentai', 'Sports', 'Mahou Shoujo', 'Horror', 'Action', 'Mystery', 'Sci-Fi']
GENRES_NORMALIZED = [x.lower().replace(' ', '_') for x in GENRES] 

class AnimeDataset(tf.data.Dataset):
    def _generator(anime_data, img_folder, training):
        anime_array = json.loads(anime_data)

        split = int(len(anime_array) // 1.5) 

        if training:
            anime_array = anime_array[0:split]
        else:
            anime_array = anime_array[split:-1]

        for i, anime_info in enumerate(anime_array):

            label = {}
            for i, genre in enumerate(GENRES):
                label[GENRES_NORMALIZED[i]] = anime_info["genres"].count(genre)
             
            cover_path = img_folder.decode('utf-8') + "/image_{}.png.jpg".format(anime_info["id"])

            if not exists(cover_path):
                print("WARNING: Picture not found {}".format(cover_path))
                continue

            if anime_info["description"] == None:
                continue
        
            image_file = open(cover_path, 'rb')
            image_raw = image_file.read()
            image = tf.io.decode_jpeg(image_raw)
            
            # Normalize the image.
            image = image / 255 - 0.5

            yield ({"img":image, "desc": tf.constant(anime_info["description"])}, label)
            #yield ({"img":image, "desc": tf.constant("[ZERO]")}, label)

 
    def __new__(self, anime_data, img_folder, training=True):
        data_spec = {"img": tf.TensorSpec(shape=(150, 100, 3), dtype=tf.float32),
             "desc": tf.TensorSpec(shape=(), dtype=tf.string)}
        labels = {}
        for genre in GENRES_NORMALIZED:
            labels[genre] = tf.TensorSpec(shape=(), dtype=tf.float32)

        spec = (data_spec, labels)
        return tf.data.Dataset.from_generator(self._generator, 
            args=(anime_data, img_folder, training),
            output_signature=spec)

def generate_corpus(anime_data):
    """
    Extract descriptions and return it as a array.
    Used for creating a word dictionary.
    """

    array = json.loads(anime_data)
    result = []
    for entry in array:
        if entry["description"] != None:
            result.append(entry["description"])

    return result
