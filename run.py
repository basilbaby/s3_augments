from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import random
import nltk
from nltk.corpus import wordnet
import argparse
import logging
from PIL import Image
import cv2
import numpy as np
import io
import base64
import time
import glob

# Set NLTK data path to tmp directory
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)

# Define upload folder and other constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif'}

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'dev'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def setup_nltk():
    """Download NLTK data to custom directory"""
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    try:
        nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True)
        nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR, quiet=True)
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

# Configure logging
logger = logging.getLogger(__name__)

def configure_logging(debug_mode):
    """Configure logging based on debug mode"""
    if debug_mode:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(set(synonyms))

def synonym_replacement(sentence, n=3):
    logger.debug(f"Starting synonym replacement for: {sentence}")
    words = sentence.split()
    replacements_made = 0
    max_attempts = len(words) * 2
    attempts = 0
    
    while replacements_made < n and attempts < max_attempts:
        word_to_replace = random.choice(words)
        attempts += 1
        
        if len(word_to_replace) <= 3 or not word_to_replace.isalnum():
            logger.debug(f"Skipping word: {word_to_replace}")
            continue
            
        synonyms = get_synonyms(word_to_replace.lower())
        logger.debug(f"Found {len(synonyms)} synonyms for '{word_to_replace}': {synonyms}")
                    
        if synonyms:
            synonym = random.choice(synonyms)
            synonym = synonym.replace('_', ' ')
            logger.debug(f"Replacing '{word_to_replace}' with '{synonym}'")
            
            if word_to_replace.istitle():
                synonym = synonym.title()
            elif word_to_replace.isupper():
                synonym = synonym.upper()
            
            words = [synonym if word.lower() == word_to_replace.lower() else word for word in words]
            replacements_made += 1
    
    result = ' '.join(words)
    logger.debug(f"Made {replacements_made} replacements")
    return result

def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def is_text_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt'}

def denoise_image(image):
    """Apply stronger denoising effect"""
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply stronger denoising with modified parameters
    # h: filter strength (higher = more denoising) (default is 10)
    # hColor: color filter strength
    # templateWindowSize: should be odd (default is 7)
    # searchWindowSize: should be odd (default is 21)
    denoised = cv2.fastNlMeansDenoisingColored(
        cv_image,
        None,
        h=20,          # Increased from default
        hColor=20,     # Increased from default
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # Add additional processing for more visible effect
    # Apply slight Gaussian blur
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Enhance edges slightly
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    denoised = cv2.filter2D(denoised, -1, kernel)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

def color_jitter(image):
    """Apply stronger color jittering effect"""
    from PIL import ImageEnhance
    
    # Increase the range of random factors for more dramatic effects
    brightness_factor = random.uniform(0.5, 1.5)    # More extreme brightness changes
    contrast_factor = random.uniform(0.5, 1.8)      # More extreme contrast changes
    saturation_factor = random.uniform(0.3, 1.7)    # More extreme saturation changes
    hue_factor = random.uniform(-0.1, 0.1)         # Add hue variation
    
    # Convert to HSV for hue adjustment
    img_hsv = image.convert('HSV')
    h, s, v = img_hsv.split()
    h = h.point(lambda x: (x + int(hue_factor * 255)) % 255)
    image = Image.merge('HSV', (h, s, v)).convert('RGB')
    
    # Apply enhanced color adjustments
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    
    # Add slight sharpening for more dramatic effect
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def save_temp_image(image):
    """Save image temporarily and return its filename"""
    if not os.path.exists('tmp/images'):
        os.makedirs('tmp/images')
    temp_filename = f"tmp/images/temp_{int(time.time())}.png"
    image.save(temp_filename)
    return temp_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logger.debug("Upload route accessed")
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filename = secure_filename(file.filename)
        
        # Debug logging
        logger.debug(f"Uploaded file: {filename}")
        logger.debug(f"File type: {file.content_type}")
        
        if not (is_image_file(filename) or is_text_file(filename)):
            return 'Unsupported file type. Please upload a .txt, .png, .jpg, .jpeg, or .gif file.', 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_info = {
            'name': filename,
            'type': file.content_type,
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        if is_image_file(filename):
            # Handle image file
            logger.debug("Processing image file")
            image = Image.open(filepath)
            # Convert image to RGB if it's in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image_data = image_to_base64(image)
            return render_template('display.html', 
                                image_data=image_data,
                                file_info=file_info,
                                is_image=True)
        elif is_text_file(filename):
            # Handle text file
            logger.debug("Processing text file")
            with open(filepath, 'r') as f:
                content = f.read()
            return render_template('display.html', 
                                text=content,
                                file_info=file_info,
                                is_image=False)
    
    return render_template('upload.html')

def format_file_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"

@app.route('/augment', methods=['POST'])
def augment():
    try:
        # Check if it's a text file
        if 'text' in request.form:
            text = request.form.get('text', '')
            if not text:
                return 'No text provided', 400
            
            # Apply synonym replacement
            augmented_text = synonym_replacement(text)
            
            file_info = {
                'name': 'Text Document',
                'type': 'text/plain',
                'size': len(text.encode('utf-8')),
                'size_formatted': format_file_size(len(text.encode('utf-8')))
            }
            
            return render_template('display.html',
                                text=text,
                                augmented_text=augmented_text,
                                is_image=False,
                                processed_type="Augmented",
                                file_info=file_info)
        
        # Handle image augmentation
        filename = request.form.get('filename')
        if not filename:
            return 'No image provided', 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return 'Image not found', 404
        
        # Create file_info dictionary
        file_info = {
            'name': filename,
            'type': 'image/png',  # or get actual mime type
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        # Open and process the image
        image = Image.open(filepath)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        augmented_image = color_jitter(image)
        augmented_filename = save_temp_image(augmented_image)
        
        with open(filepath, 'rb') as f:
            original_image_data = base64.b64encode(f.read()).decode()
        
        with open(augmented_filename, 'rb') as f:
            augmented_image_data = base64.b64encode(f.read()).decode()
        
        return render_template('display.html',
                            image_data=original_image_data,
                            processed_image=augmented_image_data,
                            is_image=True,
                            processed_type="Augmented",
                            filename=filename,
                            file_info=file_info)
    except Exception as e:
        logger.error(f"Error in image augmentation: {str(e)}", exc_info=True)
        return f"Error in image augmentation: {str(e)}", 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # Check if it's a text file
        if 'text' in request.form:
            text = request.form.get('text', '')
            if not text:
                return 'No text provided', 400
            
            # Preprocess text: remove extra spaces and convert to lowercase
            preprocessed_text = ' '.join(text.lower().split())
            
            file_info = {
                'name': 'Text Document',
                'type': 'text/plain',
                'size': len(text.encode('utf-8')),
                'size_formatted': format_file_size(len(text.encode('utf-8')))
            }
            
            return render_template('display.html',
                                text=text,
                                augmented_text=preprocessed_text,
                                is_image=False,
                                processed_type="Preprocessed",
                                file_info=file_info)
        
        # Handle image preprocessing
        filename = request.form.get('filename')
        if not filename:
            return 'No image provided', 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return 'Image not found', 404
        
        # Create file_info dictionary
        file_info = {
            'name': filename,
            'type': 'image/png',  # or get actual mime type
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        # Open and process the image
        image = Image.open(filepath)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        processed_image = denoise_image(image)
        processed_filename = save_temp_image(processed_image)
        
        with open(filepath, 'rb') as f:
            original_image_data = base64.b64encode(f.read()).decode()
        
        with open(processed_filename, 'rb') as f:
            processed_image_data = base64.b64encode(f.read()).decode()
        
        return render_template('display.html',
                            image_data=original_image_data,
                            processed_image=processed_image_data,
                            is_image=True,
                            processed_type="Preprocessed",
                            filename=filename,
                            file_info=file_info)
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}", exc_info=True)
        return f"Error in image preprocessing: {str(e)}", 500

def cleanup_temp_files():
    """Clean up temporary image files older than 1 hour"""
    current_time = time.time()
    temp_files = glob.glob('tmp/images/temp_*.png')
    
    for temp_file in temp_files:
        try:
            file_time = float(temp_file.split('_')[1].split('.')[0])
            if current_time - file_time > 3600:  # 1 hour
                os.remove(temp_file)
        except (ValueError, OSError) as e:
            logger.error(f"Error cleaning up {temp_file}: {e}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Configure logging based on debug flag
    configure_logging(args.debug)
    
    # Setup NLTK data
    logger.info("Setting up NLTK data...")
    if setup_nltk():
        logger.info(f"NLTK data downloaded successfully to {NLTK_DATA_DIR}")
    else:
        logger.error("Failed to download NLTK data")
        exit(1)

    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('tmp/images', exist_ok=True)
    
    # Clean up old temporary files
    cleanup_temp_files()
    
    logger.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=8080, debug=args.debug, threaded=True)
