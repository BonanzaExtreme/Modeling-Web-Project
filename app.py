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

            # Predict dito and display of top 3 predicted results 
            preds = model.predict(img_array)
            top_indices = np.argsort(preds[0])[::-1][:3]  # Get the top 3 predicted indices
            top_predictions = [
                {
                    'species_name': class_names[idx],
                    'probability': round(preds[0][idx] * 100, 2)
                }
                for idx in top_indices
            ]

            # Get top 1 prediction 
            species_name = class_names[top_indices[0]]
            
            # Common name and description
            common_name = species_info[species_name]["common_name"]
            description = species_info[species_name]["description"] 

            return render_template('index.html', species=species_name, common_name=common_name, description=description, top_predictions=top_predictions, img_path=filepath)

    return render_template('index.html', species=None)

if __name__ == '__main__':
    app.run(debug=True)
