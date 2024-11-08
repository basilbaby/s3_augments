from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import random
import nltk
from nltk.corpus import wordnet
import argparse
import logging

# Set NLTK data path to tmp directory
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)

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

app = Flask(__name__)

# Basic configuration
app.config['SECRET_KEY'] = 'dev'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logger.debug("Processing file upload")
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_info = {
            'name': filename,
            'type': file.content_type or 'text/plain',
            'size': os.path.getsize(filepath),
            'size_formatted': format_file_size(os.path.getsize(filepath))
        }
        
        with open(filepath, 'r') as f:
            content = f.read()
        return render_template('display.html', text=content, file_info=file_info)
    
    return render_template('upload.html')

def format_file_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"

@app.route('/augment', methods=['POST'])
def augment():
    logger.debug("Augment route accessed")
    text = request.form.get('text', '')
    if not text:
        return 'No text provided', 400
    
    try:
        augmented_text = synonym_replacement(text)
        logger.debug(f"Original text: {text}")
        logger.debug(f"Augmented text: {augmented_text}")
        
        file_info = {
            'name': 'Augmented Text',
            'type': 'text/plain',
            'size': len(text.encode('utf-8')),
            'size_formatted': format_file_size(len(text.encode('utf-8')))
        }
        
        return render_template('display.html', 
                             text=text,
                             augmented_text=augmented_text,
                             processed_type="Synonym",
                             file_info=file_info)
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}", exc_info=True)
        return f"Error in text augmentation: {str(e)}", 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    logger.debug("Preprocess route accessed")
    text = request.form.get('text', '')
    if not text:
        return 'No text provided', 400
    
    try:
        # Preprocess text: remove all extra spaces (including newlines) and convert to lowercase
        preprocessed_text = ' '.join(text.lower().split())
        logger.debug(f"Original text: {text}")
        logger.debug(f"Preprocessed text: {preprocessed_text}")
        
        file_info = {
            'name': 'Preprocessed Text',
            'type': 'text/plain',
            'size': len(text.encode('utf-8')),
            'size_formatted': format_file_size(len(text.encode('utf-8')))
        }
        
        return render_template('display.html', 
                             text=text,
                             augmented_text=preprocessed_text,
                             processed_type="Preprocessed",
                             file_info=file_info)
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return f"Error in text preprocessing: {str(e)}", 500

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
    
    logger.info("Starting Flask server...")
    app.run(host='127.0.0.1', port=8080, debug=args.debug, threaded=True)
