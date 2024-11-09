# 3D Model Viewer and Processor

A simple web application for viewing and processing 3D models in OFF format. Built with Flask and Python.

## Features

- Upload and view 3D models (OFF format)
- Real-time 3D model visualization
- Model preprocessing (centering and normalization)
- Model augmentation (random rotation and scaling)

## Requirements

- Python 3.8+
- Flask
- NumPy
- Matplotlib
- PyTorch
- Pillow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python run.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:8080
```

3. Upload a 3D model file (OFF format)
4. Use the interface buttons to:
   - View the 3D model
   - Preprocess the model (center and normalize)
   - Apply random augmentations

## File Structure
```
project_root/
├── run.py              # Main application file
├── requirements.txt    # Project dependencies
├── uploads/           # Directory for uploaded OFF files
├── tmp/              # Temporary files
│   └── images/       # Preview images
└── templates/        # HTML templates
    ├── upload.html
    └── model_view.html
```

## Supported Operations

### Preprocessing
- Center model to origin
- Normalize scale to [-1, 1]

### Augmentation
- Random rotation around Y-axis
- Random scaling

## Input Format
- Supports OFF (Object File Format) 3D model files
- Maximum file size: 16MB

## License
[Your License]