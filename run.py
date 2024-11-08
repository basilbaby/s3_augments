from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Remove any default route that might be showing "Hello, Flask!"
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("Route accessed!")  # Debug print
    if request.method == 'POST':
        print("POST request received")  # Debug print
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read the file content
            with open(filepath, 'r') as f:
                content = f.read()
            return render_template('display.html', text=content)
    
    print("Rendering upload template")  # Debug print
    return render_template('upload.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=True)
