import os
import cv2
import numpy as np
from flask import Flask, request, send_file
from io import BytesIO
import gdown

# Download model files from Google Drive
def download_model_files():
    # Replace with your actual Google Drive file IDs
    PROTOTXT_ID = '1hK56NLhwHxI61Zn3rSs7oJAu3KfCa_nI'
    MODEL_ID = '1Q8-iJjv4I7VfqTTr4VjNUpUi8ZtaagFm'
    POINTS_ID = '1evjjUeX3PN0pz0qX2Q8lZmkFq2khdjsj'

    gdown.download(f"https://drive.google.com/uc?id={PROTOTXT_ID}", 'colorization_deploy_v2.prototxt', quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", 'colorization_release_v2.caffemodel', quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={POINTS_ID}", 'pts_in_hull.npy', quiet=False)

# Download the files when the app starts
download_model_files()

# Initialize Flask app
app = Flask(__name__)

# Load model after downloading
print("Loading model...")
PROTOTXT = 'colorization_deploy_v2.prototxt'
MODEL = 'colorization_release_v2.caffemodel'
POINTS = 'pts_in_hull.npy'

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Route to colorize images
@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        if 'image' not in request.files:
            return "No file part", 400

        image_file = request.files['image']
        if image_file.filename == '':
            return "No selected file", 400

        # Read image file
        file_bytes = image_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return "Error decoding image", 400

        # Resize and process image
        small_size = (300, 300)
        image_resized = cv2.resize(image, small_size)
        scaled = image_resized.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        # Run model
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image_resized.shape[1], image_resized.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

        # Convert back to BGR
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save colorized image to memory
        is_success, buffer = cv2.imencode(".jpg", colorized)
        if not is_success:
            return "Error encoding image", 500

        return send_file(BytesIO(buffer), mimetype='image/jpeg')

    except Exception as e:
        return f"Error processing image: {str(e)}", 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
