from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Basic configuration
app.config['SECRET_KEY'] = 'dev'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Upload route accessed")  # Debug print
        if 'file' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as f:
            content = f.read()
        return render_template('display.html', text=content)
    
    print("Index route accessed")  # Debug print
    return render_template('upload.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='127.0.0.1', port=8080, debug=True, threaded=True)
