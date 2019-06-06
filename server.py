import base64
from datetime import datetime
from flask import Flask, request, render_template
import functools
from glob import glob
import json
import logging
import os
import pandas
import shutil
import tensorflow as tf
import threading
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
import cv2
from train import Denoise

denoise = Denoise(batch_size=1, img_h=400, img_w=400, img_c=1,
                          lambd=100, epoch=1, clean_path="", noised_path="",
                          save_path="",
                          learning_rate=2e-4, beta1=0., beta2=0.9, epsilon=1e-10)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

UPLOAD_FOLDER = os.getcwd()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = []
model_train = []


@app.route('/training', methods=['POST'])
def upload_training_data():
    try:
        data = request.get_data()
        jdata = json.loads(data)
        zip_contents = base64.b64decode(jdata['content'])
        DIR_NAME = jdata['pid']
        NUM_STEPS = jdata['steps']
        #NAME = jdata['name']
        #DESP = jdata['des']

        if os.path.exists(DIR_NAME):
            shutil.rmtree(os.path.join(os.getcwd(), DIR_NAME))
        zipPath = os.path.join(os.getcwd(), (DIR_NAME + '.zip'))

        with open(zipPath, 'wb') as zipFile:
            zipFile.write(zip_contents)
        zipPath = os.path.join(os.getcwd(), (DIR_NAME + '.zip'))
        PROJ_PATH = os.path.join(app.config['UPLOAD_FOLDER'], DIR_NAME)
        with ZipFile(zipPath) as zipFile:
            zipFile.extractall(PROJ_PATH)
        os.remove(os.path.join(os.getcwd(), (DIR_NAME + '.zip')))

        clean_path = os.path.join(os.getcwd(), DIR_NAME, "clean/")
        noised_path = os.path.join(os.getcwd(), DIR_NAME, "noised/")
        save_path = os.path.join(os.getcwd(), DIR_NAME, "save_para/")
        denoise = Denoise(batch_size=1, img_h=400, img_w=400, img_c=1,
                          lambd=100, epoch=NUM_STEPS, clean_path=clean_path, noised_path=noised_path,
                          save_path=save_path,
                          learning_rate=2e-4, beta1=0., beta2=0.9, epsilon=1e-10)
        denoise.train()
        return "Data uploaded successfully"
    except Exception as e:
        print("upload", e)
        return "Unable to upload data"

# @app.route('/status', methods=['GET', 'POST'])
# def cur_status():
#     try:
#         data = request.get_data()
#         data = json.loads(data)
#         pid = data['pid']
#         conn = connect_sql()
#         with conn:
#             status = get_status(conn, pid)
#         conn.close()
#         return status
#     except Exception as e:
#         print(e)
#         return "Invalid Project id"


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    import time
    try:
        data = request.get_data()
        jdata = json.loads(data)
        nparr = np.fromstring(base64.b64decode(jdata['image']), np.uint8)
        DIR_NAME = jdata['pid']
        global denoise
        if os.path.exists(DIR_NAME):
            if DIR_NAME not in model:
                model.append(DIR_NAME)
                clean_path = os.path.join(os.getcwd(), DIR_NAME, "clean/")
                noised_path = os.path.join(os.getcwd(), DIR_NAME, "noised/")
                save_path = os.path.join(os.getcwd(), DIR_NAME, "save_para/")
                denoise = Denoise(batch_size=1, img_h=400, img_w=400, img_c=1,
                                  lambd=100, epoch=1, clean_path=clean_path, noised_path=noised_path,
                                  save_path=save_path,
                                  learning_rate=2e-4, beta1=0., beta2=0.9, epsilon=1e-10)
                denoise.load(save_path)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            start = time.time()
            img_str = denoise.test(img)
            end = time.time()
            print(end - start)
            data = {}
            data["data"] = str(img_str)
            final_data = json.dumps(data)
            return final_data
    except Exception as e:
        print(e)
        return "The project ID is either invalid or model is under training"


@app.route('/models', methods=['GET', 'POST'])
def get_pre_trained_model():
    try:
        conn = connect_sql()
        cur = conn.cursor()
        cur.execute("SELECT * FROM ModelMap")
        data = cur.fetchall()
        l =[]
        col = ("PID", "Model", "MAP", "Loss", "Name", "Description", "Classes", "Status")
        l.append(col)
        for d in data:
            l.append(d)
        json_file = json.dumps(l)
        return json_file
    except Exception as e:
        print(e)
        return "unable to reach database"


if __name__ == '__main__':
    app.secret_key = "djkhkhd"
    app.run(host="192.168.3.34", port=3006, threaded=True, debug=True)