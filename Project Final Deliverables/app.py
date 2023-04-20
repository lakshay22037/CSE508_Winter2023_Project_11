import random
import string
from flask import Flask, render_template, request, send_file, url_for
from Code import predict
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pickle
import cv2
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import time
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
from tensorflow.python.ops.numpy_ops import np_config
import requests
import json


# Load the inverted Index 
with open('static/clustering_index.pkl', 'rb') as file:
    # Call load method to deserialize
    index1 = pickle.load(file)

# Load efficientnet architecture for extracting embeddings of query images
import efficientnet.keras as efn 
efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=False, classes=100, input_shape=(224, 224, 3))

model = Sequential()
model.add(efnb0)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

# load pretrained weights
model.load_weights('static/best_weight.h5')

# Build encoder model from efficientnet
newModel = Model(inputs=model.inputs, outputs=model.layers[1].output)

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define the allowed extensions for the file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define a function to check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route to handle the file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)
        # If the file is valid
        if file and allowed_file(file.filename):
            # Save the file with a secure filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Redirect to the result page
            return redirect(url_for('result', filename=filename))
    # Render the file upload form
    return render_template('index.html')

# Define a route to handle the image retrieval and related audio description
@app.route('/result', methods=['GET'])
def result():
    # Get the filename of the uploaded image
    filename = request.args.get('filename')
    # Save the uploaded file to the static/uploads directory
    file = request.files['file']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Get the path to the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load the query image and get its embedding
    query_img = image.load_img(file_path, target_size=(224, 224))
    query_img = image.img_to_array(query_img)
    query_img /= 255
    query_img = np.expand_dims(query_img, axis=0)
    query_embedding = newModel(query_img)

    # compute the Euclidean distances between the query embedding and all the embeddings in the inverted index
    distances = []
    for i in range(len(index)):
        distances.append(cosine(query_embedding.reshape(-1),index[i][0].reshape(-1)))

    # find the closest term
    closest_term = np.argmin(distances)

    # get the posting list for the closest term and compute the distances between the query embedding and the embeddings in the posting list
    posting_list = index[closest_term][1]
    posting_distances = []
    for posting in posting_list:
        dist = np.linalg.norm(posting[1] - query_embedding)
        posting_distances.append((posting[0], dist))

    # sort the posting list in ascending order of the distances to the query embedding
    posting_distances.sort(key=lambda x: x[1])

    # return the 5 closest images
    posting_distances = posting_distances[:5]
    closest_images = []
    # closest_audio = []
    query_string=[]
    for id, dist in posting_distances:
        path = 'static/train_images/image' + str(int(id)) + '.png'
        # closest_images.append(path)
        closest_images.append(url_for('static/train_images', filename=str(str(int(id)) + '.png')))
        print(path)
        labels, audio_description = predict(path)

         # Generate a random query string to force the browser to refresh the audio file
        qu_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        query_string.append(qu_string)
    # Pass the URL of the uploaded image to the HTML page
    image_url = url_for('static', filename='image.jpg')

    # render the result template with the query image, closest images and audio descriptions
    return render_template('result.html', query_image=file_path, closest_images=closest_images, labels=labels, audio_description=audio_description, query_string=qu_string,image_url=image_url)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/serve_audio1')
def serve_audio1():
    try:
        filename = 'static/description1.mp3'
        return send_file(filename, mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Error: Audio file not found"
    
@app.route('/serve_audio2')
def serve_audio2():
    try:
        filename = 'static/description2.mp3'
        return send_file(filename, mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Error: Audio file not found"

@app.route('/serve_audio3')
def serve_audio3():
    try:
        filename = 'static/description3.mp3'
        return send_file(filename, mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Error: Audio file not found"

@app.route('/serve_audio4')
def serve_audio4():
    try:
        filename = 'static/description4.mp3'
        return send_file(filename, mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Error: Audio file not found"

@app.route('/serve_audio5')
def serve_audio5():
    try:
        filename = 'static/description5.mp3'
        return send_file(filename, mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Error: Audio file not found"

@app.route('/audio')
def serve_audio():
    filename = 'static/description.mp3'
    return send_file(filename, mimetype='audio/mpeg')


@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded image file from the HTML form
    file = request.files['image']

    # Save the file to disk
    file.save('static/image.jpg')

    # Load the query image and get its embedding
    query_img = image.load_img("static/image.jpg", target_size=(224, 224))
    query_img = image.img_to_array(query_img)
    query_img /= 255
    query_img = np.expand_dims(query_img, axis=0)
    query_embedding = newModel(query_img)

    # compute the Euclidean distances between the query embedding and all the embeddings in the inverted index
    distances = []
    for i in range(len(index1)):
        np_config.enable_numpy_behavior()
        distances.append(cosine(query_embedding.reshape(-1),index1[i][0].reshape(-1)))

    # find the closest term
    closest_term = np.argmin(distances)

    # get the posting list for the closest term and compute the distances between the query embedding and the embeddings in the posting list
    posting_list = index1[closest_term][1]
    posting_distances = []
    for posting in posting_list:
        dist = np.linalg.norm(posting[1] - query_embedding)
        posting_distances.append((posting[0], dist))

    # sort the posting list in ascending order of the distances to the query embedding
    posting_distances.sort(key=lambda x: x[1])

    # return the 5 closest images
    posting_distances = posting_distances[:5]
    closest_images = []
    # closest_audio = []
    query_string=[]
    relevant_labels=[]
    relevant_audio_desc=[]
    rel_path=[]
    # static_url = url_for('static')
    for id, dist in posting_distances:
        path = 'static/train_images/train_images/image' + str(int(id)) + '.png'
        rel_path.append(path)
        closest_images.append(url_for('static', filename='train_images/train_images/image'+str(int(id)) + '.png'))

    relevant_labels, relevant_audio_desc = predict(rel_path,1)
    print(rel_path)

    # Get the object labels and audio description using the predict() function from Code.py
    labels, audio_description = predict('static/image.jpg',0)

    # Generate a random query string to force the browser to refresh the audio file
    query_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # Pass the URL of the uploaded image to the HTML page
    image_url = url_for('static', filename='image.jpg')

    # Render the labels, audio description, and query string on the HTML page
    return render_template('result.html', labels=labels, audio_description=audio_description, query_string=query_string, image_url=image_url,closest_images=closest_images, relevant_labels=relevant_labels,relevant_audio_desc=relevant_audio_desc)

if __name__ == '__main__':
    app.run(debug=True)