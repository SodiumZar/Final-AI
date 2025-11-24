# System Architecture - Retina Analysis with GenAI

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                     (templates/index.html)                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Upload     │  │  Display     │  │   Results    │        │
│  │   Section    │→ │  Processing  │→ │   Section    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK APPLICATION                          │
│                         (app.py)                                │
│                                                                 │
│  1. Receive uploaded image                                     │
│  2. Save to static/uploads/                                    │
│  3. Call prediction pipeline                                   │
│  4. Generate visualizations                                    │
│  5. Calculate metrics                                          │
│  6. Get AI analysis                                            │
│  7. Render results template                                    │
└─────────────────────────────────────────────────────────────────┘
        ↓              ↓              ↓              ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PREDICTION  │ │   UTILS      │ │  OVERLAY     │ │  GENAI       │
│  (predict.py)│ │  (utils.py)  │ │  CREATION    │ │  ANALYZER    │
│              │ │              │ │              │ │              │
│ • Load model │ │• Density     │ │• Load image  │ │• Initialize  │
│ • Preprocess │ │• Tortuosity  │ │• Create mask │ │  client      │
│ • Segment    │ │• Analysis    │ │• Blend       │ │• Upload img  │
│ • Threshold  │ │              │ │• Save        │ │• Get analysis│
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT FILES                            │
│                                                                 │
│  static/uploads/        static/predictions/  static/overlays/  │
│  ├── image.jpg          ├── image.jpg        ├── overlay_*.jpg │
│  └── ...                └── ...              └── ...           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
INPUT: Retina Fundus Image (512x512 RGB)
  │
  ├─→ [U-Net Model] ──→ Binary Mask (512x512)
  │                       │
  │                       ├─→ [Save] → static/predictions/
  │                       │
  │                       ├─→ [Vessel Density] → 15.2%
  │                       │
  │                       ├─→ [Tortuosity] → 1.18
  │                       │
  │                       └─→ [Overlay] → static/overlays/
  │
  └─→ [GenAI Analysis] ──→ Medical Summary
        ↑
        │ (metrics)
        └─ Density + Tortuosity

OUTPUT: 5 Results
  1. Segmented Mask Image
  2. Overlay Visualization
  3. Vessel Density Metric
  4. Tortuosity Score
  5. AI Medical Summary
```

## Component Interaction

```
┌────────────────────────────────────────────────────────────┐
│                    WEB BROWSER                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Original │  │  Mask    │  │ Overlay  │  │ Metrics  │  │
│  │  Image   │  │  Image   │  │  Image   │  │  Cards   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│  └────────────── AI Summary Section ──────────────────┘   │
└────────────────────────────────────────────────────────────┘
                         ↑
                         │ HTTP Response
                         │
┌────────────────────────────────────────────────────────────┐
│                    FLASK SERVER                            │
│                                                            │
│  app.py (Router)                                          │
│    │                                                       │
│    ├─→ predict.py (Model)                                │
│    │     └─→ model.py (U-Net)                            │
│    │                                                       │
│    ├─→ utils.py (Analysis)                               │
│    │     ├─→ create_overlay()                            │
│    │     ├─→ calculate_vessel_density()                  │
│    │     └─→ calculate_tortuosity()                      │
│    │                                                       │
│    └─→ retina_analyzer.py (GenAI)                        │
│          └─→ Google Gemini API                           │
└────────────────────────────────────────────────────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND                             │
│  • HTML5 (Jinja2 Templates)                            │
│  • TailwindCSS (Styling)                               │
│  • JavaScript (File Upload)                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   BACKEND (Flask)                       │
│  • Python 3.12                                         │
│  • Flask 3.1.2 (Web Framework)                         │
│  • Werkzeug (WSGI)                                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              DEEP LEARNING (PyTorch)                    │
│  • PyTorch 2.9.0 (Framework)                           │
│  • U-Net Model (Segmentation)                          │
│  • CUDA/CPU Support                                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           IMAGE PROCESSING                              │
│  • OpenCV (cv2) - Image I/O, Processing                │
│  • NumPy - Array Operations                            │
│  • PIL/Pillow - Image Manipulation                     │
│  • scikit-image - Skeletonization                      │
│  • scipy - Scientific Computing                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              AI ANALYSIS (GenAI)                        │
│  • Google GenAI SDK                                    │
│  • Gemini 2.0 Flash Model                              │
│  • Natural Language Processing                         │
└─────────────────────────────────────────────────────────┘
```

## Processing Pipeline

```
Step 1: Image Upload
  └─→ Validate format
  └─→ Save to uploads/

Step 2: Segmentation (2-3s)
  └─→ Load U-Net model
  └─→ Preprocess image
  └─→ Forward pass
  └─→ Apply threshold
  └─→ Save mask

Step 3: Overlay Creation (1s)
  └─→ Load original
  └─→ Highlight vessels
  └─→ Blend images
  └─→ Save overlay

Step 4: Metric Calculation (1-2s)
  └─→ Count vessel pixels
  └─→ Calculate density
  └─→ Skeletonize mask
  └─→ Compute tortuosity

Step 5: AI Analysis (5-10s)
  └─→ Upload to GenAI
  └─→ Generate prompt
  └─→ Get analysis
  └─→ Format output

Step 6: Display Results
  └─→ Render template
  └─→ Show 3 images
  └─→ Display metrics
  └─→ Present AI summary
```

## File Structure

```
Final Ai/
├── app.py                     # Main Flask application
├── predict.py                 # Segmentation logic
├── model.py                   # U-Net architecture
├── train.py                   # Training script
├── utils.py                   # Utility functions
│                              # • Overlay creation
│                              # • Density calculation
│                              # • Tortuosity analysis
├── retina_analyzer.py         # GenAI integration
├── ai_client.py              # GenAI test client
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── test_system.py            # System tests
│
├── templates/
│   └── index.html            # Web interface
│
├── static/
│   ├── uploads/              # User uploads
│   ├── predictions/          # Segmentation masks
│   └── overlays/             # Overlay images
│
├── files/
│   └── checkpoint.pth        # Model weights
│
├── Data/
│   ├── train/                # Training data
│   │   ├── image/
│   │   └── mask/
│   └── test/                 # Test data
│       ├── image/
│       └── mask/
│
└── Documentation/
    ├── README.md             # Full documentation
    ├── QUICK_START.md        # User guide
    └── IMPLEMENTATION_SUMMARY.md  # This file
```

## API Integration

```
┌─────────────────────────────────────────────────────────┐
│              Google Gemini API                          │
│                                                         │
│  Request:                                               │
│  ┌─────────────────────────────────────────────────┐  │
│  │ POST /v1/models/gemini-2.0-flash-exp:generate  │  │
│  │                                                 │  │
│  │ Headers:                                        │  │
│  │   Authorization: Bearer [API_KEY]              │  │
│  │                                                 │  │
│  │ Body:                                           │  │
│  │   - Original retina image (uploaded)           │  │
│  │   - Detailed medical prompt                    │  │
│  │   - Quantitative metrics (density, tortuosity) │  │
│  └─────────────────────────────────────────────────┘  │
│                         ↓                               │
│  Response:                                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │ {                                               │  │
│  │   "text": "**Overall Assessment**\n..."        │  │
│  │ }                                               │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Scalability Considerations

```
Current: Single Image Processing
  └─→ ~15-20 seconds per image

Future Enhancements:
  ├─→ Batch Processing (multiple images)
  ├─→ Async Processing (background tasks)
  ├─→ Caching (repeated analysis)
  ├─→ Database (store results)
  └─→ API Endpoints (RESTful API)
```

---

**Created**: November 16, 2025
**Version**: 1.0
**Status**: Production Ready ✅
