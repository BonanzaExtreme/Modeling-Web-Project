import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import json

model = load_model('models/efficientnetb4_model.h5')
with open('data/species_info.json') as f:
    species_info = json.load(f)

class_names = list(species_info.keys()) 

IMG_SIZE = (380, 380)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filepath = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(filepath)

            # Dito ipepreprocess yung image
            img = image.load_img(filepath, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict dito 
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            species_name = class_names[class_idx]
            description = species_info[species_name]

            return render_template('index.html', species=species_name, description=description, img_path=filepath)

    return render_template('index.html', species=None)

if __name__ == '__main__':
    app.run(debug=True)
