# Retina Blood Vessel Segmentation with GenAI Analysis

## Overview
This application uses deep learning to segment blood vessels in retina fundus images and leverages Google's GenAI (Gemini) to provide comprehensive medical analysis.

## Features

### 1. **Segmented Image (Mask)**
- Binary segmentation mask showing blood vessels
- U-Net based deep learning model
- High accuracy vessel detection

### 2. **Overlay Visualization**
- Original image with highlighted blood vessels in red
- Semi-transparent overlay for better visualization
- Easy identification of vessel patterns

### 3. **Vessel Density**
- Percentage of vessel pixels in the image
- Normal range: 10-20%
- Automatic classification (Normal/Outside typical range)

### 4. **Tortuosity Score**
- Measure of vessel curvature
- Calculated using skeletonization and path analysis
- Normal range: 1.0-1.3
- Higher values may indicate diabetic or hypertensive changes

### 5. **AI-Powered Medical Summary**
- Comprehensive analysis using Google Gemini AI
- Professional medical assessment
- Clinical observations and recommendations
- Context-aware insights based on quantitative metrics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the model checkpoint:
- Place `checkpoint.pth` in the `files/` directory

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload a retina fundus image

4. View the comprehensive analysis including:
   - Original image
   - Segmented mask
   - Overlay visualization
   - Vessel density and tortuosity metrics
   - AI-generated medical summary

## Project Structure

```
Final Ai/
├── app.py                  # Flask application
├── predict.py              # Prediction logic
├── model.py                # U-Net model architecture
├── utils.py                # Utility functions (overlay, density, tortuosity)
├── retina_analyzer.py      # GenAI integration
├── ai_client.py            # GenAI client example
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── static/
│   ├── uploads/           # Uploaded images
│   ├── predictions/       # Segmented masks
│   └── overlays/          # Overlay visualizations
├── files/
│   └── checkpoint.pth     # Model weights
└── Data/
    ├── train/             # Training data
    └── test/              # Test data
```

## Technical Details

### Vessel Density Calculation
- Formula: `(vessel_pixels / total_pixels) × 100`
- Provides percentage of image occupied by vessels

### Tortuosity Score Calculation
1. Skeletonize the vessel mask to get centerlines
2. For each vessel segment:
   - Calculate actual path length
   - Calculate Euclidean distance (straight line)
   - Tortuosity = path_length / euclidean_distance
3. Average across all vessel segments

### GenAI Integration
- Uses Google Gemini 2.0 Flash model
- Analyzes original image with quantitative metrics
- Provides medical-grade assessment
- Includes clinical observations and recommendations

## API Key

The application uses Google GenAI. The API key is currently hardcoded in `retina_analyzer.py`. For production use, consider:
- Using environment variables
- Implementing secure key storage
- Rate limiting and error handling

## Disclaimer

⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. The AI-generated analysis should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## Dependencies

Key packages:
- Flask (Web framework)
- PyTorch (Deep learning)
- OpenCV (Image processing)
- scikit-image (Advanced image processing)
- scipy (Scientific computing)
- google-genai (AI analysis)
- numpy, pillow (Image manipulation)

## Future Enhancements

- [ ] Multiple image upload
- [ ] Batch processing
- [ ] Export reports as PDF
- [ ] Historical tracking
- [ ] User authentication
- [ ] Database integration
- [ ] Advanced vessel analysis (bifurcations, vessel width)
- [ ] Comparison with previous scans

## License

This project is for educational purposes.

## Contributors

Developed with AI assistance for retinal image analysis.
