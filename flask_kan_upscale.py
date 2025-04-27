from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Kiểm tra thiết bị (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa model KAN_SR
class KAN_SR(nn.Module):
    def __init__(self):
        super(KAN_SR, self).__init__()
        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 64 * 64)

    def forward(self, x):
        x = x.view(-1, 8 * 8)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output trong khoảng [0,1]
        x = x.view(-1, 1, 64, 64)  # Reshape thành ảnh
        return x

# Load model đã train
model = KAN_SR().to(device)
model.load_state_dict(torch.load("kan_sr_model.pth", map_location=device))
model.eval()

# Transform ảnh
transform_low = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor()])
transform_high = transforms.Compose([transforms.Resize((64, 64))])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Xử lý ảnh với mô hình
            with Image.open(filepath).convert("L") as img:  # Chuyển về grayscale
                low_res = transform_low(img).unsqueeze(0).to(device)  # Resize về 8x8
                
                start_time = time.time()  # Bắt đầu đo thời gian
                with torch.no_grad():
                    upscaled = model(low_res).cpu().squeeze()  # Phóng đại lên 64x64
                end_time = time.time()  # Kết thúc đo thời gian
                
                processing_time = end_time - start_time  # Tính thời gian xử lý

                # Chuyển tensor thành ảnh
                upscaled_img = transforms.ToPILImage()(upscaled)
                upscaled_filename = "upscaled_" + filename
                upscaled_filepath = os.path.join(app.config['UPLOAD_FOLDER'], upscaled_filename)
                upscaled_img.save(upscaled_filepath)

            return f'''
            <!doctype html>
            <html>
            <head>
                <title>KAN Image Upscaling</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body class="container py-5">
                <h1 class="text-center mb-4">KAN Uploaded Image and Upscaled Version</h1>
                <p class="text-center">File: {filename}</p>
                <p class="text-center">Processing Time: {processing_time:.4f} seconds</p>
                <div class="row justify-content-center">
                    <div class="col-md-4 text-center">
                        <h3>Original Image</h3>
                        <img style="height: 300px; aspect-ratio: 1;" src="/uploads/{filename}" class="img-fluid rounded shadow">
                    </div>
                    <div class="col-md-4 text-center">
                        <h3>Upscaled Image (64x64)</h3>
                        <img style="height: 300px; aspect-ratio: 1;" src="/uploads/{upscaled_filename}" class="img-fluid rounded shadow">
                    </div>
                </div>
                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary">Upload Another Image</a>
                </div>
            </body>
            </html>
            '''
    
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Upload Image</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            #drop-area {
                width: 100%;
                max-width: 400px;
                height: 200px;
                border: 2px dashed #ccc;
                text-align: center;
                padding: 20px;
                margin: 20px auto;
                background: #f8f9fa;
            }
        </style>
    </head>
    <body class="container py-5 text-center">
        <h1>Upload an Image</h1>
        <div id="drop-area" class="rounded">
            <p>Drop files here</p>
        </div>
        <form method=post enctype=multipart/form-data class="mt-3">
          <input type=file name=file id="file-input" class="form-control mb-2">
          <input type=submit value=Upload class="btn btn-success">
        </form>
        <script>
            let dropArea = document.getElementById('drop-area');
            let fileInput = document.getElementById('file-input');
            
            dropArea.addEventListener('dragover', (event) => {
                event.preventDefault();
                dropArea.style.background = '#e9ecef';
            });
            dropArea.addEventListener('dragleave', () => {
                dropArea.style.background = '#f8f9fa';
            });
            dropArea.addEventListener('drop', (event) => {
                event.preventDefault();
                dropArea.style.background = '#f8f9fa';
                let files = event.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                }
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5002)
