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

def create_model_preview(vertices, faces, size=(400, 400)):
    """Create a 2D preview with enhanced visualization"""
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Convert to numpy if needed
    verts = vertices.numpy() if torch.is_tensor(vertices) else vertices
    faces_np = faces.numpy() if torch.is_tensor(faces) else faces
    
    # Calculate bounds with padding
    min_coords = verts.min(axis=0)
    max_coords = verts.max(axis=0)
    center = (min_coords + max_coords) / 2
    
    # Add more padding for better visibility
    scale = min(size) * 0.7 / max(max_coords - min_coords)
    
    # Project vertices with enhanced visibility
    points_2d = []
    for v in verts:
        x = (v[0] - center[0]) * scale + size[0] / 2
        y = (v[1] - center[1]) * scale + size[1] / 2
        points_2d.append((x, y))
    
    # Draw faces with enhanced styling
    for face in faces_np:
        # Draw filled face
        draw.polygon([points_2d[i] for i in face], 
                    fill='#e0e0e0',
                    outline='black')
        
        # Draw edges with thicker lines
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            draw.line([points_2d[v1], points_2d[v2]], 
                     fill='black', 
                     width=2)
    
    return img

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
                # 3D Model handling (existing code)
                vertices, faces = load_off_file(filepath)
                preview_img = create_model_preview(vertices, faces)
                
                # Save preview
                preview_path = os.path.join(app.config['TEMP_FOLDER'], f'preview_{filename}.png')
                preview_img.save(preview_path)
                
                with open(preview_path, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode()
                
                file_info = {
                    'name': filename,
                    'type': '3D Model (OFF)',
                    'size': f'{os.path.getsize(filepath)} B',
                    'vertices': vertices.shape[0],
                    'faces': faces.shape[0]
                }
                
                model_info = {'preview': preview_data}
                return render_template('model_view.html', file_info=file_info, model_info=model_info)
            
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
    """Augment the 3D model with multiple transformations"""
    # Convert to numpy arrays
    vertices_np = vertices.numpy() if torch.is_tensor(vertices) else np.array(vertices)
    faces_np = faces.numpy() if torch.is_tensor(faces) else np.array(faces)
    
    # Step 1: Center the model first
    center = np.mean(vertices_np, axis=0)
    vertices_centered = vertices_np - center
    
    # Step 2: Apply random rotations on multiple axes
    # Random angles between -45 and 45 degrees
    theta_x = np.random.uniform(-np.pi/4, np.pi/4)
    theta_y = np.random.uniform(-np.pi/4, np.pi/4)
    theta_z = np.random.uniform(-np.pi/4, np.pi/4)
    
    # Rotation matrices
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Apply all rotations
    vertices_rotated = vertices_centered @ rot_x @ rot_y @ rot_z
    
    # Step 3: Add random scaling (between 0.8 and 1.2)
    scale = np.random.uniform(0.8, 1.2)
    vertices_scaled = vertices_rotated * scale
    
    # Step 4: Add small random translation
    translation = np.random.uniform(-0.2, 0.2, size=3)
    vertices_final = vertices_scaled + translation
    
    logger.debug(f"Augmentation applied: scale={scale:.2f}, " +
                f"rotations(deg)=({np.degrees(theta_x):.1f}, {np.degrees(theta_y):.1f}, {np.degrees(theta_z):.1f})")
    
    return torch.tensor(vertices_final, dtype=torch.float32), torch.tensor(faces_np, dtype=torch.long)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = UPLOAD_DIR / secure_filename(filename)
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Load the model
        vertices, faces = load_off_file(filepath)
        
        # Preprocess the model
        vertices_processed, faces_processed = preprocess_model(vertices, faces)
        
        # Create preview
        img = create_model_preview(vertices_processed, faces_processed)
        thumb_path = TEMP_DIR / f'preview_processed_{int(time.time())}.png'
        img.save(thumb_path)
        
        with open(thumb_path, 'rb') as f:
            thumbnail_data = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'preview': thumbnail_data,
            'vertices': vertices_processed.shape[0],
            'faces': faces_processed.shape[0]
        })
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/augment', methods=['POST'])
def augment():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        filepath = UPLOAD_DIR / secure_filename(filename)
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        # Load the model
        vertices, faces = load_off_file(filepath)
        
        # Augment the model
        vertices_augmented, faces_augmented = augment_model(vertices, faces)
        
        # Create preview
        img = create_model_preview(vertices_augmented, faces_augmented)
        thumb_path = TEMP_DIR / f'preview_augmented_{int(time.time())}.png'
        img.save(thumb_path)
        
        with open(thumb_path, 'rb') as f:
            thumbnail_data = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'preview': thumbnail_data,
            'vertices': vertices_augmented.shape[0],
            'faces': faces_augmented.shape[0]
        })
        
    except Exception as e:
        logger.error(f"Error in augmentation: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
