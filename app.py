from flask import Flask, render_template, request

from keras.preprocessing.image import load_image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = VGG16()

@app.route('/', methods=['GET'])
def render_templat_app():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./result/" + imagefile.filename
    imagefile.save(image_path)

    image = load_image(image_path, target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], image.shape[2])
    yhat = model.predict(image)
    label = decode_prediction(yhat)
    label = label[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', predictions=classification)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
