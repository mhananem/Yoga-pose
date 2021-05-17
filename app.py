from flask import Flask, render_template,request
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


app = Flask(__name__)

classifier = load_model('Final_Yoga_pose_classifier.h5')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/", methods=["POST"])
def predict():
    imagefile = request.files["imagefile"]
    image_path ="./static/" + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    result = classifier.predict(images,batch_size=10) #return array
    if result[0][0] == 1:
        pose = 'Downdog' #predictions
    elif result[0][1] == 1:
        pose = 'Goddess'
    elif result[0][2] == 1:
        pose = 'Plank'
    elif result[0][3] == 1:
        pose = 'Tree'
    elif result[0][4] == 1:
        pose = 'Warrior'
    
    return render_template("index.html", prediction=pose,)

if __name__ == "__main__":
    app.run(debug=True)
