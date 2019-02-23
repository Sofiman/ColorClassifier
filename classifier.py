from keras.models import load_model
import numpy as np
from colour import Color

model = load_model('training.h5')

line = ""
print("Write 'quit' to stop")
while True:
    line = input(":")
    if line.lower() == "quit":
        break
    else:
        try:
            c = Color(line)
            color = np.array([c.rgb])
            r = model.predict(color)
            dark = round(r[0][0])
            print(line, "is a", "Dark" if dark else "Light", "color")
            
        except Exception:
            print("An error occurred while trying to classify the color")
