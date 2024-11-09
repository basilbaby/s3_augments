# DataForge

A Flask-based web application that provides preprocessing and augmentation capabilities for text, image, and audio files.

## Features

### Text Processing
1. **Preprocessing**
   - Whitespace Removal
   - Lowercase Conversion
   - Standardizes text formatting

2. **Augmentation**
   - Synonym Replacement (up to 3 words)
   - Maintains original word casing
   - Preserves text structure

### Image Processing
1. **Preprocessing (Denoising)**
   - Noise Reduction
   - Edge Enhancement
   - Gaussian Blur
   - Image Smoothing

2. **Augmentation**
   - Color Jittering
   - Brightness Adjustment (±50%)
   - Contrast Enhancement (±80%)
   - Saturation Modification (±70%)
   - Hue Variation (±10%)
   - Image Sharpening

### Audio Processing
1. **Preprocessing**
   - Noise Reduction
   - Audio Enhancement
   - Signal Preprocessing

2. **Augmentation**
   - Speed Modification (20% faster)
   - Pitch Preservation
   - Audio Quality Maintenance

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python run.py
   ```

2. Access the web interface:
   - Open your browser and go to `http://127.0.0.1:8080`
   - The interface supports both text (.txt) and image (.png, .jpg, .jpeg, .gif) files

3. Processing Files:
   - Click "Choose File" to select your input file
   - Click "Upload and Process" to load your file
   - Use "Preprocess" for noise reduction/text standardization
   - Use "Augmentation" for synonym replacement/color enhancement

## File Support

### Text Files
- Supported format: `.txt`
- Maximum file size: 16MB
- UTF-8 encoding recommended

### Image Files
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.gif`
- Maximum file size: 16MB
- Both RGB and RGBA images supported

### Audio Files
- Supported formats: `.wav`, `.mp3`
- Maximum file size: 16MB
- Both mono and stereo audio supported

## Technical Details

- Built with Flask web framework
- Uses NLTK for text processing
- OpenCV and PIL for image processing
- Temporary files are automatically cleaned up after 1 hour
- Debug mode available with `python run.py --debug`

## Directory Structure

```
├── run.py              # Main application file
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── upload.html    # File upload interface
│   └── display.html   # Results display
├── uploads/           # Uploaded files directory
└── tmp/              # Temporary files directory
    └── images/       # Processed images storage
```

## Dependencies

- Flask
- NLTK
- OpenCV (cv2)
- Pillow (PIL)
- NumPy

## Notes

- Processed files are temporarily stored and automatically cleaned up
- Original files are preserved
- All processing is done server-side
- Interface is responsive and works on mobile devices