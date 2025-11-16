from google import genai
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
try:
    from config import GENAI_API_KEY, GENAI_MODEL
except ImportError:
    GENAI_API_KEY = "AIzaSyDcTvsNocVzRQKp8-5L00b6XNRY2S1ZnGc"
    GENAI_MODEL = "gemini-2.0-flash-exp"

class RetinaAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the GenAI client for retina image analysis"""
        self.api_key = api_key or GENAI_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model = GENAI_MODEL
    
    def image_to_base64(self, image_array):
        """Convert numpy array to base64 string for API"""
        # Convert to PIL Image
        if len(image_array.shape) == 2:
            # Grayscale
            image = Image.fromarray((image_array * 255).astype(np.uint8))
        else:
            # Color image
            image = Image.fromarray(image_array.astype(np.uint8))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def analyze_retina_image(self, original_image_path, mask, vessel_metrics):
        """
        Analyze retina image using GenAI and provide comprehensive summary
        Args:
            original_image_path: Path to the original retina image
            mask: Binary segmentation mask
            vessel_metrics: Dictionary containing vessel density and tortuosity
        Returns:
            analysis: Detailed analysis summary from GenAI
        """
        # Read the original image
        original_image = cv2.imread(original_image_path)
        
        # Create the prompt for GenAI
        prompt = f"""You are an expert ophthalmologist analyzing a retina fundus image. 
        
Based on the retina image and its blood vessel segmentation, provide a comprehensive medical analysis.

**Quantitative Metrics:**
- Vessel Density: {vessel_metrics['vessel_density']}%
- Tortuosity Score: {vessel_metrics['tortuosity_score']} (1.0 = straight, >1.5 = tortuous)
- Total Vessel Pixels: {vessel_metrics['total_vessel_pixels']}
- Image Size: {vessel_metrics['image_size']}

**Please provide:**

1. **Overall Assessment**: Brief evaluation of the retinal blood vessel pattern (2-3 sentences)

2. **Vessel Characteristics**: 
   - Comment on vessel density (normal range: 10-20%)
   - Comment on tortuosity (normal range: 1.0-1.3)
   - Vessel distribution pattern

3. **Clinical Observations**:
   - Any notable patterns or anomalies
   - Potential indicators of conditions (diabetic retinopathy, hypertension, etc.)
   - Areas of concern if any

4. **Recommendations**:
   - Suggested follow-up actions
   - Whether further examination is needed

Keep the response professional, clear, and medically accurate. Format using bullet points and clear sections."""

        try:
            # Upload the image file
            uploaded_file = self.client.files.upload(path=original_image_path)
            
            # Generate analysis using GenAI
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    uploaded_file,
                    prompt
                ]
            )
            
            # Clean up uploaded file
            self.client.files.delete(name=uploaded_file.name)
            
            return response.text
            
        except Exception as e:
            print(f"Error in GenAI analysis: {str(e)}")
            return self._generate_fallback_summary(vessel_metrics)
    
    def _generate_fallback_summary(self, vessel_metrics):
        """Generate a basic summary if GenAI fails"""
        density = vessel_metrics['vessel_density']
        tortuosity = vessel_metrics['tortuosity_score']
        
        summary = f"""
**RETINA ANALYSIS SUMMARY**

**Overall Assessment:**
The retinal blood vessel segmentation has been successfully completed. The analysis reveals a vessel density of {density}% with a tortuosity score of {tortuosity}.

**Vessel Characteristics:**
- Vessel Density: {density}% {"(Within normal range)" if 10 <= density <= 20 else "(Outside typical range)"}
- Tortuosity Score: {tortuosity} {"(Normal)" if tortuosity < 1.3 else "(Elevated - may indicate vessel abnormalities)"}
- Total Vessel Pixels: {vessel_metrics['total_vessel_pixels']}

**Clinical Observations:**
- {"The vessel density appears normal for a healthy retina." if 10 <= density <= 20 else "The vessel density is outside the typical range, which may warrant further investigation."}
- {"Vessel paths show normal curvature." if tortuosity < 1.3 else "Elevated tortuosity may indicate hypertensive or diabetic changes."}

**Recommendations:**
- Regular monitoring recommended
- Consult with an ophthalmologist for comprehensive eye examination
- This automated analysis should be confirmed by a medical professional

*Note: This is an automated analysis. Please consult a qualified healthcare professional for medical diagnosis.*
        """
        
        return summary.strip()
