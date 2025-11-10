from flask import Flask, request, render_template, send_from_directory
from predict import predict
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Create directories if they don't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            if not os.path.exists(app.config['PREDICTION_FOLDER']):
                os.makedirs(app.config['PREDICTION_FOLDER'])

            # Save the uploaded file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Make prediction
            predicted_mask = predict(image_path)

            # Save the predicted mask
            predicted_mask_color = cv2.cvtColor(predicted_mask * 255, cv2.COLOR_GRAY2BGR)
            prediction_path = os.path.join(app.config['PREDICTION_FOLDER'], file.filename)
            cv2.imwrite(prediction_path, predicted_mask_color)

            return render_template('index.html', uploaded_image=image_path, predicted_image=prediction_path)

    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
