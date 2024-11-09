import os
import time
import torch
import base64
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Define directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
TEMP_DIR = BASE_DIR / 'tmp' / 'images'

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# App configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=str(UPLOAD_DIR),
    TEMP_FOLDER=str(TEMP_DIR),
    DEBUG=True
)

def format_file_size(size_in_bytes):
    """Format file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"

def load_off_file(filepath):
    """Load OFF file and return vertices and faces"""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Skip header
        header = lines[0].strip()
        if header != 'OFF':
            raise ValueError('Invalid OFF file format')
            
        # Read counts
        counts = lines[1].strip().split()
        num_vertices = int(counts[0])
        num_faces = int(counts[1])
        
        # Read vertices
        for i in range(num_vertices):
            vertex = [float(x) for x in lines[i + 2].strip().split()]
            vertices.append(vertex)
            
        # Read faces
        for i in range(num_faces):
            face = [int(x) for x in lines[i + 2 + num_vertices].strip().split()[1:]]
            faces.append(face)
    
    return torch.tensor(vertices), torch.tensor(faces)

def create_model_preview(vertices, faces):
    """Create a preview image of the 3D model"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the model
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                    triangles=faces, color='lightgray', edgecolor='black')
    
    # Add coordinate axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add grid
    ax.grid(True)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set view angle
    ax.view_init(elev=20, azim=45)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Convert buffer to PIL Image
    img = Image.open(buf)
    return img.convert('RGB')

def process_3d_file(filepath, filename):
    """Process a 3D file and create visualization"""
    try:
        # Load the model
        vertices, faces = load_off_file(filepath)
        
        # Create visualization
        img = create_model_preview(vertices, faces)
        
        # Save visualization
        thumb_path = TEMP_DIR / f'preview_{int(time.time())}.png'
        img.save(thumb_path)
        
        with open(thumb_path, 'rb') as f:
            thumbnail_data = base64.b64encode(f.read()).decode()
        
        # Get file stats
        file_size = os.path.getsize(filepath)
        
        file_info = {
            'name': filename,
            'type': '3D Model (OFF)',
            'size': f"{file_size} B",
            'vertices': vertices.shape[0],
            'faces': faces.shape[0]
        }
        
        return render_template('model_view.html',
                             file_info=file_info,
                             model_info={'preview': thumbnail_data})
                             
    except Exception as e:
        logger.error(f"Error processing 3D file: {str(e)}")
        return f"Error processing 3D file: {str(e)}", 400

def process_text_file(filepath, filename):
    """Process a text file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        file_size = os.path.getsize(filepath)
        
        file_info = {
            'name': filename,
            'type': 'Text',
            'size': f"{file_size} B",
            'content': content
        }
        
        return render_template('text_view.html', file_info=file_info)
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        return f"Error processing text file: {str(e)}", 400

def process_image_file(filepath, filename):
    """Process an image file"""
    try:
        file_size = os.path.getsize(filepath)
        
        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        
        file_info = {
            'name': filename,
            'type': 'Image',
            'size': f"{file_size} B",
            'data': image_data
        }
        
        return render_template('image_view.html', file_info=file_info)
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        return f"Error processing image file: {str(e)}", 400

def process_audio_file(filepath, filename):
    """Process an audio file"""
    try:
        file_size = os.path.getsize(filepath)
        
        file_info = {
            'name': filename,
            'type': 'Audio',
            'size': f"{file_size} B",
            'path': str(filepath)
        }
        
        return render_template('audio_view.html', file_info=file_info)
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return f"Error processing audio file: {str(e)}", 400

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Determine file type and render appropriate template
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension == '.off':
                # 3D Model handling
                vertices, faces = load_off_file(filepath)
                preview_img = create_model_preview(vertices, faces)
                
                # Save preview
                preview_path = os.path.join('tmp', 'images', f'preview_{filename}.png')
                preview_img.save(preview_path)
                
                with open(preview_path, 'rb') as f:
                    model_data = base64.b64encode(f.read()).decode()
                
                file_info = {
                    'name': filename,
                    'type': '3D Model (OFF)',
                    'size': format_file_size(os.path.getsize(filepath)),
                    'vertices': vertices.shape[0],
                    'faces': faces.shape[0]
                }
                
                return render_template('model_view.html',
                                     file_info=file_info,
                                     model_data=model_data)
            
            elif file_extension in ['.txt']:
                # Text file handling
                with open(filepath, 'r') as f:
                    content = f.read()
                file_info = {
                    'name': filename,
                    'type': 'Text',
                    'size': f'{os.path.getsize(filepath)} B',
                    'content': content
                }
                return render_template('text_view.html', file_info=file_info)
                
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                # Image file handling
                img = Image.open(filepath)
                file_info = {
                    'name': filename,
                    'type': 'Image',
                    'size': f'{os.path.getsize(filepath)} B',
                    'dimensions': f'{img.size[0]}x{img.size[1]}',
                    'format': img.format
                }
                return render_template('image_view.html', file_info=file_info, image_path=filename)
                
            elif file_extension in ['.mp3', '.wav']:
                # Audio file handling
                file_info = {
                    'name': filename,
                    'type': 'Audio',
                    'size': f'{os.path.getsize(filepath)} B'
                }
                return render_template('audio_view.html', file_info=file_info, audio_path=filename)
    
    # GET request - show upload form
    return render_template('upload.html')

# Add route for serving uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def preprocess_model(vertices, faces):
    """Preprocess the 3D model with clearly visible transformations"""
    # Convert to numpy arrays
    vertices_np = vertices.numpy() if torch.is_tensor(vertices) else np.array(vertices)
    faces_np = faces.numpy() if torch.is_tensor(faces) else np.array(faces)
    
    # Step 1: Center the model
    center = np.mean(vertices_np, axis=0)
    vertices_centered = vertices_np - center
    
    # Step 2: Normalize scale
    max_distance = np.max(np.abs(vertices_centered))
    vertices_normalized = vertices_centered / max_distance
    
    # Step 3: Apply more dramatic transformations
    # Rotate 60 degrees around Y
    theta_y = np.pi/3
    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Rotate 45 degrees around X
    theta_x = np.pi/4
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Apply rotations
    vertices_rotated = vertices_normalized @ rot_y @ rot_x
    
    # Step 4: Add more dramatic scaling and shifting
    vertices_final = vertices_rotated * 0.6  # Scale down more
    vertices_final[:, 1] += 0.4  # Shift up more
    
    logger.debug(f"Preprocessing applied: scale=0.6, shift=0.4, rotation_y={np.degrees(theta_y)}°, rotation_x={np.degrees(theta_x)}°")
    
    return torch.tensor(vertices_final, dtype=torch.float32), torch.tensor(faces_np, dtype=torch.long)

def augment_model(vertices, faces):
    """Apply random augmentations to the 3D model"""
    # Convert to numpy if needed
    vertices_np = vertices.numpy() if torch.is_tensor(vertices) else vertices
    faces_np = faces.numpy() if torch.is_tensor(faces) else faces
    
    # Random rotation angle (in radians)
    angle = np.random.uniform(0, 2 * np.pi)
    
    # Random scaling factor
    scale = np.random.uniform(0.8, 1.2)
    
    # Create rotation matrix around Y-axis
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # Apply rotation
    vertices_rotated = np.dot(vertices_np, rotation_matrix)
    
    # Apply scaling
    vertices_scaled = vertices_rotated * scale
    
    # Return transformed vertices, faces, and transformation parameters
    return (torch.tensor(vertices_scaled), 
            torch.tensor(faces_np),
            {'angle': angle, 'scale': scale})

@app.route('/preprocess', methods=['POST'])
def preprocess():
    logger.info("Preprocessing route called")
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No file provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        # Load the model
        vertices, faces = load_off_file(filepath)
        
        # Apply preprocessing
        processed_vertices, processed_faces = preprocess_model(vertices, faces)
        
        # Create preview images
        preview_img = create_model_preview(processed_vertices, processed_faces)
        preview_path = os.path.join('tmp', 'images', f'preview_processed_{filename}.png')
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        preview_img.save(preview_path)
        
        # Create original preview for comparison
        original_preview = create_model_preview(vertices, faces)
        original_path = os.path.join('tmp', 'images', f'preview_original_{filename}.png')
        original_preview.save(original_path)
        
        with open(preview_path, 'rb') as f:
            processed_data = base64.b64encode(f.read()).decode()
            
        with open(original_path, 'rb') as f:
            original_data = base64.b64encode(f.read()).decode()
        
        preprocessing_info = {
            'operations': [
                'Centered model to origin',
                'Normalized scale to [-1, 1]',
                'Rotated 45° around Y-axis'
            ]
        }
        
        return render_template('model_view.html',
                            file_info={
                                'name': filename,
                                'type': '3D Model (OFF)',
                                'size': format_file_size(os.path.getsize(filepath)),
                                'vertices': vertices.shape[0],
                                'faces': faces.shape[0]
                            },
                            model_data=original_data,
                            processed_model=processed_data,
                            preprocessing_info=preprocessing_info,
                            processed_type="Preprocessed Model")
                            
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error during preprocessing: {str(e)}'}), 500

@app.route('/augment', methods=['POST'])
def augment():
    logger.info("Augmentation route called")
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No file provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        # Load the model
        vertices, faces = load_off_file(filepath)
        
        # Apply augmentation
        augmented_vertices, augmented_faces, aug_params = augment_model(vertices, faces)
        
        # Create preview images
        preview_img = create_model_preview(augmented_vertices, augmented_faces)
        preview_path = os.path.join('tmp', 'images', f'preview_augmented_{filename}.png')
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        preview_img.save(preview_path)
        
        # Create original preview for comparison
        original_preview = create_model_preview(vertices, faces)
        original_path = os.path.join('tmp', 'images', f'preview_original_{filename}.png')
        original_preview.save(original_path)
        
        with open(preview_path, 'rb') as f:
            augmented_data = base64.b64encode(f.read()).decode()
            
        with open(original_path, 'rb') as f:
            original_data = base64.b64encode(f.read()).decode()
        
        augmentation_info = {
            'operations': [
                f'Random rotation: {np.round(np.degrees(aug_params["angle"]), 2)}°',
                f'Random scaling: {np.round(aug_params["scale"], 2)}x',
                'Position preserved'
            ]
        }
        
        return render_template('model_view.html',
                            file_info={
                                'name': filename,
                                'type': '3D Model (OFF)',
                                'size': format_file_size(os.path.getsize(filepath)),
                                'vertices': vertices.shape[0],
                                'faces': faces.shape[0]
                            },
                            model_data=original_data,
                            processed_model=augmented_data,
                            preprocessing_info=augmentation_info,
                            processed_type="Augmented Model")
                            
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error during augmentation'}), 500

@app.route('/preprocess_text', methods=['POST'])
def preprocess_text():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Preprocessing steps
        processed_content = content.lower()  # Convert to lowercase
        processed_content = ' '.join(processed_content.split())  # Normalize whitespace
        
        return jsonify({
            'success': True,
            'content': processed_content
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/augment_text', methods=['POST'])
def augment_text():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Simple word-by-word synonym replacement
        words = content.split()
        augmented_words = []
        
        for word in words:
            # Get synsets for the word
            synsets = wordnet.synsets(word)
            
            # If synsets exist, get the first synonym that's different from the original word
            if synsets:
                for synset in synsets:
                    found_synonym = False
                    for lemma in synset.lemmas():
                        if lemma.name() != word:
                            augmented_words.append(lemma.name())
                            found_synonym = True
                            break
                    if found_synonym:
                        break
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
        
        augmented_content = ' '.join(augmented_words)
        
        return jsonify({
            'success': True,
            'content': augmented_content
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
