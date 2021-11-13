import random
import json

DATA_FILE = "cleaned.json"
d = open(DATA_FILE)
raw_data = json.loads(d.read())

random.shuffle(raw_data)

result_f = open("cleaned_shuffled.json", "w")
result_f.write(json.dumps(raw_data))

