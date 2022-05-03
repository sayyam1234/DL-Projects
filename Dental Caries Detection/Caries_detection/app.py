import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import cv2



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (748, 512)
img_size = (748, 512)
model =  tf.keras.models.load_model('model/model.h5',compile=False)
UPLOAD_FOLDER = 'uploads'
label_dict={0:'caries', 1:'healthy'}

def preprocess(img):

	img=np.array(img)
	img_batch = np.expand_dims(img, axis=0)
	if(img.ndim==4):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	#gray = gray/255
	
	reshaped = resized.reshape(1,img_size,img_size,3)
	reshaped = reshaped.astype('float32')
	reshaped = reshaped / 255
	print(resized)
	reshaped = transpose(reshaped)
	print(reshaped.ndim)
	return reshaped
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    img  = load_img(file, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = img / 255	
    print(img)
    #img = transpose(img)
    #test_image=preprocess(img)
    probs = model.predict(img)
    print(probs)
    result=np.argmax(probs,axis=1)[0]
    accuracy=float(np.max(probs,axis=1)[0])
    print(result)
    output = str(label_dict[result])+" Accuracy"+str(accuracy) 
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('index.html', label='', imagesource='')

output="null"
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
        else:
            print("output")
    return render_template("index.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
