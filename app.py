from flask import Flask, request, render_template, send_from_directory
from predict import predict
from utils import create_overlay, analyze_vessels
from retina_analyzer import RetinaAnalyzer
import os
import cv2
import numpy as np
import markdown

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
OVERLAY_FOLDER = 'static/overlays'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['OVERLAY_FOLDER'] = OVERLAY_FOLDER

# Initialize the retina analyzer
analyzer = RetinaAnalyzer()

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
            if not os.path.exists(app.config['OVERLAY_FOLDER']):
                os.makedirs(app.config['OVERLAY_FOLDER'])

            # Save the uploaded file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Make prediction (supports fallback returning dict)
            result = predict(image_path)
            warning_msg = None
            if isinstance(result, dict):
                predicted_mask = result.get('mask')
                warning_msg = result.get('warning')
            else:
                predicted_mask = result

            # Save the predicted mask
            predicted_mask_color = cv2.cvtColor(predicted_mask * 255, cv2.COLOR_GRAY2BGR)
            prediction_path = os.path.join(app.config['PREDICTION_FOLDER'], file.filename)
            cv2.imwrite(prediction_path, predicted_mask_color)

            # Create overlay visualization
            original_image = cv2.imread(image_path)
            overlay = create_overlay(original_image, predicted_mask)
            overlay_path = os.path.join(app.config['OVERLAY_FOLDER'], f"overlay_{file.filename}")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Analyze vessel characteristics
            vessel_metrics = analyze_vessels(predicted_mask)

            # Get GenAI analysis
            ai_summary = analyzer.analyze_retina_image(image_path, predicted_mask, vessel_metrics)

            # Convert Markdown to HTML with rich formatting
            ai_summary_html = markdown.markdown(
                ai_summary,
                extensions=['extra', 'nl2br', 'sane_lists', 'tables']
            )

            return render_template('index.html', 
                                uploaded_image=image_path, 
                                predicted_image=prediction_path,
                                overlay_image=overlay_path,
                                vessel_density=vessel_metrics['vessel_density'],
                                tortuosity_score=vessel_metrics['tortuosity_score'],
                                ai_summary=ai_summary_html,
                                vessel_metrics=vessel_metrics,
                                warning_msg=warning_msg)

    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
