from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import base64
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# App configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='uploads',
    SECRET_KEY='dev'
)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs(os.path.join('tmp', 'images'), exist_ok=True)

ALLOWED_EXTENSIONS = {'off'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def preprocess_model(vertices, faces):
    """Apply preprocessing to the 3D model"""
    # Center the model
    center = vertices.mean(dim=0)
    vertices = vertices - center
    
    # Scale to unit cube
    scale = vertices.abs().max()
    vertices = vertices / scale
    
    return vertices, faces

def augment_model(vertices, faces):
    """Apply random augmentations to the 3D model"""
    # Random rotation angle (in radians)
    angle = np.random.uniform(0, 2 * np.pi)
    
    # Random scaling factor
    scale = np.random.uniform(0.8, 1.2)
    
    # Create rotation matrix around Y-axis
    rotation_matrix = torch.tensor([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=vertices.dtype)
    
    # Apply rotation
    vertices_rotated = torch.mm(vertices, rotation_matrix)
    
    # Apply scaling
    vertices_scaled = vertices_rotated * scale
    
    return vertices_scaled, faces, {'angle': angle, 'scale': scale}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and create preview
            vertices, faces = load_off_file(filepath)
            preview_img = create_model_preview(vertices, faces)
            preview_path = os.path.join('tmp', 'images', f'preview_{filename}.png')
            preview_img.save(preview_path)
            
            with open(preview_path, 'rb') as f:
                model_data = base64.b64encode(f.read()).decode()
            
            return render_template('model_view.html',
                                file_info={
                                    'name': filename,
                                    'type': '3D Model (OFF)',
                                    'size': format_file_size(os.path.getsize(filepath)),
                                    'vertices': vertices.shape[0],
                                    'faces': faces.shape[0]
                                },
                                model_data=model_data)
    
    return render_template('upload.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No file provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        vertices, faces = load_off_file(filepath)
        processed_vertices, processed_faces = preprocess_model(vertices, faces)
        
        preview_img = create_model_preview(processed_vertices, processed_faces)
        preview_path = os.path.join('tmp', 'images', f'preview_processed_{filename}.png')
        preview_img.save(preview_path)
        
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
                'Normalized scale to [-1, 1]'
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
        return jsonify({'error': 'Error during preprocessing'}), 500

@app.route('/augment', methods=['POST'])
def augment():
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No file provided'}), 400
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        vertices, faces = load_off_file(filepath)
        augmented_vertices, augmented_faces, aug_params = augment_model(vertices, faces)
        
        preview_img = create_model_preview(augmented_vertices, augmented_faces)
        preview_path = os.path.join('tmp', 'images', f'preview_augmented_{filename}.png')
        preview_img.save(preview_path)
        
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
                f'Random scaling: {np.round(aug_params["scale"], 2)}x'
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

if __name__ == '__main__':
    app.run(debug=True, port=8080)
