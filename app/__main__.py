from flask import Flask
from pathlib import Path
import os
import shutil
from fastai.vision import *


app = Flask(__name__)

@app.route('/forecast')
def predict():
    path = Path('./models')
    new_learn_cycle = load_new_learn_cycle(path, 'network.pkl')
    
    files = os.listdir(src_folder)
    src_folder = './ml_images/'
    

    predictions = {}

    for file_name in files:
        img = open_image(os.path.join(src_folder, file_name))
        pred_class,pred_idx,outputs = new_learn_cycle.predict(img)
        proc = str(outputs[pred_idx])[9:11]
        predictions[file_name] = [pred_class, proc]
        if int(proc) >= 91:
            shutil.move(os.path.join(src_folder, file_name), f'./exact_predictions/{pred_class}_{proc}.jpg')
        else:
            shutil.move(os.path.join(src_folder, file_name), f'./inexact_predictions/{pred_class}_{proc}.jpg')

    return {'success': "forecast is done"}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500)