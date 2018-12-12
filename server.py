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

from train import Denoise


# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
# create formatter and add it to the handlers
# jkfh
formatter = logging.Formatter('%(asctime)s - %(message)s')

UPLOAD_FOLDER = os.getcwd()

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'xml'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = []
model_train = []


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

@app.route('/status', methods=['GET', 'POST'])
def cur_status():
    try:
        data = request.get_data()
        data = json.loads(data)
        pid = data['pid']
        conn = connect_sql()
        with conn:
            status = get_status(conn, pid)
        conn.close()
        return status
    except Exception as e:
        print(e)
        return "Invalid Project id"


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    import time
    start = time.time()
    try:
        data = request.get_data()
        jdata = json.loads(data)
        nparr = np.fromstring(base64.b64decode(jdata['image']), np.uint8)
        DIR_NAME = jdata['pid']
        clean_path = os.path.join(os.getcwd(), DIR_NAME, "clean/")
        noised_path = os.path.join(os.getcwd(), DIR_NAME, "noised/")
        save_path = os.path.join(os.getcwd(), DIR_NAME, "save_para/")
        denoise = Denoise(batch_size=1, img_h=400, img_w=400, img_c=1,
                          lambd=100, epoch=1, clean_path=clean_path, noised_path=noised_path,
                          save_path=save_path,
                          learning_rate=2e-4, beta1=0., beta2=0.9, epsilon=1e-10)
        if os.path.exists(DIR_NAME):
            if DIR_NAME not in model:
                model.append(DIR_NAME)
                ckpt_path = os.path.join(os.getcwd(), DIR_NAME, 'inference_graph', 'frozen_inference_graph.pb')
                global sess, detection_boxes, detection_scores, detection_classes, \
                    num_detections, image_tensor
                sess, detection_boxes, detection_scores, detection_classes, \
                num_detections, image_tensor = load_model(ckpt_path)

            json_file = detect_img(pid, nparr, sess, detection_boxes, detection_scores,
                                   detection_classes, num_detections, image_tensor)
            end = time.time()
            print(end - start)
            return json_file
        denoise.test(nparr, save_path)
        with open('output.png', "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        data = {}
        data["data"] = encoded_string.decode('ascii')
        final_data = json.dumps(data)
        # final_data = json.loads(final_data)
        # print(final_data)
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





def _check_log(DIR_NAME, NUM_STEPS):
    t = datetime.strptime("27/04/2060 16:30", "%d/%m/%Y %H:%M")
    while True:
        conn = connect_sql()
        if os.path.exists(os.path.join(os.getcwd(), (DIR_NAME + '/tensorflow.log'))) and conn:
            with open(os.path.join(os.getcwd(), (DIR_NAME + '/tensorflow.log')), "r") as tfl:
                try:
                    lastline = list(tfl)[-1].split(" - ")
                    if lastline[1].split(" ")[0] == "global":
                        s = lastline[1]
                        with conn:
                            update_status(conn, (s,DIR_NAME))
                            conn.commit()
                        if int(lastline[1].split(" ")[2].split(":")[0]) == int(NUM_STEPS):
                            t = lastline[0].split(",")[-2]
                            t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                    elif lastline[1].split(" ")[0] == "Stopping" or datetime.now() > t:
                        t = lastline[0].split(",")[-2]
                        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                        s = "Finished Training"
                        with conn:
                            update_status(conn, (s, DIR_NAME))
                            conn.commit()
                        break
                    tfl.close()

                except Exception as e:
                    # print("checklog:", e)
                    s = "Training..."
                    with conn:
                        update_status(conn, (s, DIR_NAME))
                        conn.commit()
        conn.close()


if __name__ == '__main__':
    app.secret_key = "djkhkhd"
    app.run(host="192.168.3.34", port=3006 , threaded=True, debug=True)