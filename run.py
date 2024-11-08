from flask import Flask, render_template, request, url_for, send_from_directory
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
from pydub import AudioSegment
import librosa
import soundfile as sf
import tempfile
import subprocess

# Set NLTK data path to tmp directory
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)

# Define upload folder and other constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav'}

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

def is_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav'}

def audio_to_base64(audio_path):
    with open(audio_path, 'rb') as audio_file:
        return base64.b64encode(audio_file.read()).decode()

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        # Try default path first
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Try common installation paths
        common_paths = [
            '/usr/local/bin/ffmpeg',
            '/usr/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',
            '/opt/local/bin/ffmpeg'
        ]
        
        for ffmpeg_path in common_paths:
            try:
                if os.path.exists(ffmpeg_path):
                    # Update pydub's path to ffmpeg
                    from pydub import AudioSegment
                    AudioSegment.converter = ffmpeg_path
                    probe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
                    if os.path.exists(probe_path):
                        AudioSegment.ffprobe = probe_path
                    return True
            except Exception as e:
                logger.debug(f"Failed to use ffmpeg at {ffmpeg_path}: {e}")
                continue
        
        logger.error("ffmpeg/ffprobe not found in common locations")
        return False

def setup_audio_processing():
    """Configure audio processing settings"""
    try:
        from pydub import AudioSegment
        # Try to find ffmpeg in common locations
        common_paths = [
            '/usr/local/bin/ffmpeg',
            '/usr/bin/ffmpeg',
            '/opt/homebrew/bin/ffmpeg',
            '/opt/local/bin/ffmpeg'
        ]
        
        ffmpeg_path = None
        for path in common_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break
        
        if ffmpeg_path:
            AudioSegment.converter = ffmpeg_path
            probe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            if os.path.exists(probe_path):
                AudioSegment.ffprobe = probe_path
            logger.info(f"Using ffmpeg from: {ffmpeg_path}")
            return True
        else:
            logger.error("Could not find ffmpeg installation")
            return False
    except Exception as e:
        logger.error(f"Error setting up audio processing: {e}")
        return False

def denoise_audio(audio_path):
    """Remove noise from audio file"""
    try:
        logger.debug(f"Loading audio file: {audio_path}")
        
        # Check ffmpeg installation
        if not check_ffmpeg():
            raise RuntimeError("ffmpeg is not installed. Please install ffmpeg first.")
        
        # Create temp directory if it doesn't exist
        os.makedirs('tmp/audio', exist_ok=True)
        
        # Convert MP3 to WAV if necessary
        temp_wav = None
        if audio_path.lower().endswith('.mp3'):
            temp_wav = f"tmp/audio/temp_{int(time.time())}_original.wav"
            audio = AudioSegment.from_mp3(audio_path)
            audio.export(temp_wav, format='wav')
            input_path = temp_wav
        else:
            input_path = audio_path
        
        # Load the audio file
        y, sr = librosa.load(input_path)
        
        # Apply noise reduction
        y_denoised = librosa.effects.preemphasis(y)
        
        # Save to temporary file
        temp_path = f"tmp/audio/temp_{int(time.time())}_denoised.wav"
        logger.debug(f"Saving denoised audio to: {temp_path}")
        sf.write(temp_path, y_denoised, sr)
        
        # Clean up temporary WAV file if created
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error in audio denoising: {str(e)}", exc_info=True)
        raise

def speed_up_audio(audio_path):
    """Increase audio speed by 20%"""
    try:
        logger.debug(f"Loading audio file for speed up: {audio_path}")
        
        # Check ffmpeg installation
        if not check_ffmpeg():
            raise RuntimeError("ffmpeg is not installed. Please install ffmpeg first.")
        
        # Create temp directory if it doesn't exist
        os.makedirs('tmp/audio', exist_ok=True)
        
        # Convert MP3 to WAV if necessary
        temp_wav = None
        if audio_path.lower().endswith('.mp3'):
            temp_wav = f"tmp/audio/temp_{int(time.time())}_original.wav"
            audio = AudioSegment.from_mp3(audio_path)
            audio.export(temp_wav, format='wav')
            input_path = temp_wav
        else:
            input_path = audio_path
        
        # Load audio using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Speed up by 20%
        faster_audio = audio.speedup(playback_speed=1.2)
        
        # Save to temporary file
        temp_path = f"tmp/audio/temp_{int(time.time())}_faster.wav"
        logger.debug(f"Saving speed-up audio to: {temp_path}")
        faster_audio.export(temp_path, format='wav')
        
        # Clean up temporary WAV file if created
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error in audio speed up: {str(e)}", exc_info=True)
        raise

def is_allowed_file(filename):
    """Check if the file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
        if not is_allowed_file(filename):
            return 'Unsupported file type. Please upload a .txt, .png, .jpg, .jpeg, .gif, .mp3, or .wav file.', 400
        
        # Create file info dictionary
        file_info = {
            'name': filename,
            'type': file.content_type,
            'size': 0,  # Will be updated after saving
            'size_formatted': '0 B'
        }
        
        # Handle different file types
        if is_text_file(filename):
            # Process text file
            text = file.read().decode('utf-8')
            file_info['size'] = len(text.encode('utf-8'))
            file_info['size_formatted'] = format_file_size(file_info['size'])
            return render_template('display.html',
                                text=text,
                                is_image=False,
                                is_audio=False,
                                file_info=file_info)
        
        elif is_image_file(filename):
            # Process image file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_info['size'] = os.path.getsize(filepath)
            file_info['size_formatted'] = format_file_size(file_info['size'])
            
            with open(filepath, 'rb') as f:
                image_bytes = f.read()
                image_data = base64.b64encode(image_bytes).decode()
            
            return render_template('display.html',
                                image_data=image_data,
                                is_image=True,
                                is_audio=False,
                                file_info=file_info)
        
        elif is_audio_file(filename):
            # Process audio file
            logger.debug("Processing audio file")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_info['size'] = os.path.getsize(filepath)
            file_info['size_formatted'] = format_file_size(file_info['size'])
            
            audio_data = audio_to_base64(filepath)
            return render_template('display.html',
                                audio_data=audio_data,
                                is_image=False,
                                is_audio=True,
                                file_info=file_info)
    
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
                                is_audio=False,
                                processed_type="Augmented",
                                file_info=file_info)
        
        # Get filename from form
        filename = request.form.get('filename')
        if not filename:
            return 'No file provided', 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return 'File not found', 404
        
        # Create file_info dictionary
        file_info = {
            'name': filename,
            'type': 'image/png' if is_image_file(filename) else 'audio/wav',
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        # Handle audio files
        if is_audio_file(filename):
            logger.debug("Augmenting audio file")
            try:
                # Process audio
                augmented_path = speed_up_audio(filepath)
                
                return render_template('display.html',
                                    audio_data=audio_to_base64(filepath),
                                    processed_audio=audio_to_base64(augmented_path),
                                    is_image=False,
                                    is_audio=True,
                                    processed_type="Augmented",
                                    file_info=file_info)
            except Exception as e:
                logger.error(f"Error in audio augmentation: {str(e)}", exc_info=True)
                return f"Error in audio augmentation: {str(e)}", 500
        
        # Handle image files
        elif is_image_file(filename):
            logger.debug("Augmenting image file")
            try:
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
                                    is_audio=False,
                                    processed_type="Augmented",
                                    file_info=file_info)
            except Exception as e:
                logger.error(f"Error in image augmentation: {str(e)}", exc_info=True)
                return f"Error in image augmentation: {str(e)}", 500
        
        else:
            return 'Unsupported file type', 400
            
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}", exc_info=True)
        return f"Error in augmentation: {str(e)}", 500

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
                                is_audio=False,
                                processed_type="Preprocessed",
                                file_info=file_info)
        
        # Get filename from form
        filename = request.form.get('filename')
        if not filename:
            return 'No file provided', 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return 'File not found', 404
        
        # Create file_info dictionary
        file_info = {
            'name': filename,
            'type': 'image/png' if is_image_file(filename) else 'audio/wav',
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        # Handle audio files
        if is_audio_file(filename):
            logger.debug("Preprocessing audio file")
            try:
                # Process audio
                processed_path = denoise_audio(filepath)
                
                return render_template('display.html',
                                    audio_data=audio_to_base64(filepath),
                                    processed_audio=audio_to_base64(processed_path),
                                    is_image=False,
                                    is_audio=True,
                                    processed_type="Preprocessed",
                                    file_info=file_info)
            except Exception as e:
                logger.error(f"Error in audio preprocessing: {str(e)}", exc_info=True)
                return f"Error in audio preprocessing: {str(e)}", 500
        
        # Handle image files
        elif is_image_file(filename):
            logger.debug("Preprocessing image file")
            try:
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
                                    is_audio=False,
                                    processed_type="Preprocessed",
                                    file_info=file_info)
            except Exception as e:
                logger.error(f"Error in image preprocessing: {str(e)}", exc_info=True)
                return f"Error in image preprocessing: {str(e)}", 500
        
        else:
            return 'Unsupported file type', 400
            
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return f"Error in preprocessing: {str(e)}", 500

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    current_time = time.time()
    
    # Clean up image files
    temp_images = glob.glob('tmp/images/temp_*.png')
    for temp_file in temp_images:
        try:
            file_time = float(temp_file.split('_')[1].split('.')[0])
            if current_time - file_time > 3600:  # 1 hour
                os.remove(temp_file)
        except (ValueError, OSError) as e:
            logger.error(f"Error cleaning up image {temp_file}: {e}")
    
    # Clean up audio files
    temp_audio = glob.glob('tmp/audio/temp_*.wav')
    for temp_file in temp_audio:
        try:
            file_time = float(temp_file.split('_')[1].split('.')[0])
            if current_time - file_time > 3600:  # 1 hour
                os.remove(temp_file)
        except (ValueError, OSError) as e:
            logger.error(f"Error cleaning up audio {temp_file}: {e}")

# Add this route for favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Configure logging based on debug flag
    configure_logging(args.debug)
    
    # Check ffmpeg installation
    if not check_ffmpeg():
        logger.error("ffmpeg is not installed. Please install ffmpeg to process audio files.")
        print("Error: ffmpeg is not installed. Please install ffmpeg to process audio files.")
        print("Installation instructions:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
        exit(1)
    
    # Setup NLTK data
    logger.info("Setting up NLTK data...")
    if setup_nltk():
        logger.info(f"NLTK data downloaded successfully to {NLTK_DATA_DIR}")
    else:
        logger.error("Failed to download NLTK data")
        exit(1)

    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('tmp/images', exist_ok=True)
    os.makedirs('tmp/audio', exist_ok=True)
    
    # Clean up old temporary files
    cleanup_temp_files()
    
    logger.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=8080, debug=args.debug, threaded=True)
