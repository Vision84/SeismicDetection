import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import io
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pandas as pd

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Output: 32 x 300 x 1000
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32 x 150 x 500
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64 x 150 x 500
        self.fc1 = nn.Linear(64 * 150 * 500 // 4, 128)  # Adjust this value if necessary
        self.fc2 = nn.Linear(128, 2)  # Output: 2 classes (noise or not noise)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 150 * 500 // 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("Defined CNN.")

# Load the saved model weights
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set model to evaluation mode

print("Model loaded!")

###########


from flask import (Flask, render_template, redirect, request, jsonify, url_for, g, Blueprint, Response)
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads/'

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/', methods=['GET'])
    def index():
        return render_template("index.html")

    @app.route('/api/v1/processFile/', methods=['POST'])
    def process():
        try:
            # Expect a CSV file to be sent
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            data_cat = pd.read_csv(request.files['file'])

            def fun(data_cat):
                mi = data_cat['time_rel(sec)'].tolist().index(min(data_cat['time_rel(sec)'].tolist()))
                ma = data_cat['time_rel(sec)'].tolist().index(max(data_cat['time_rel(sec)'].tolist()))
                mii = min(data_cat['velocity(m/s)'].tolist())
                maa = max(data_cat['velocity(m/s)'].tolist())
                s = 10000
                # Capture the result of thething() and return it
                return thething(mi, ma, mii, maa, s, n=[])

            def thething(mi, ma, mii, maa, s, n):
                for i in range(mi, ma, s):
                    c = i + s
                    if i + s <= ma:
                        csv_times = np.array(data_cat['time_rel(sec)'].tolist()[i:c])
                        csv_data = np.array(data_cat['velocity(m/s)'].tolist()[i:c])
                        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
                        ax.plot(csv_times, csv_data)
                        ax.set_xlim([min(csv_times), max(csv_times)])
                        ax.set_ylim([mii, maa])
                        ax.set_ylabel('Velocity (m/s)')
                        ax.set_xlabel('Time (s)')
                        ax.set_title('j', fontweight='bold')
                        fig.canvas.draw()
                        image = np.array(fig.canvas.renderer.buffer_rgba())
                        plt.close(fig)
                        transform = transforms.Compose([
                            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Ensure 3 channels (RGB)
                            transforms.Resize((300, 1000)),
                            transforms.ToTensor()
                        ])
                        image_tensor = transform(Image.fromarray(image))  # Prepare image for model
                        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                        with torch.no_grad():
                            output = model(image_tensor)
                            _, pred = torch.max(output.data, 1)
                        if pred.item() == 1:
                            o=data_cat['time_rel(sec)'].tolist()[i]
                            p=data_cat['time_rel(sec)'].tolist()[c]
                            s = s // 2
                            n.append(o)
                            if s==0:
                                break
                            thething(int(o), int(p), mii, maa, s, n)
                return list(n)
            
            # Call the fun function with the uploaded data
            result = fun(data_cat)

            # Plot the trace!
            csv_times = np.array(data_cat['time_rel(sec)'].tolist())
            csv_data = np.array(data_cat['velocity(m/s)'].tolist())
            fig,ax = plt.subplots(1,1,figsize=(10,3))
            ax.plot(csv_times,csv_data)
            ax.set_xlim([min(csv_times),max(csv_times)])
            ax.set_ylim([min(csv_data),max(csv_data)])
            ax.set_ylabel('Velocity (m/s)')
            ax.set_xlabel('Time (s)')
            ax.set_title('j', fontweight='bold')
            for i in result:
                arrival_line = ax.axvline(x=i, c='red', label='Abs. Arrival')
                ax.legend(handles=[arrival_line])

            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return Response(output.getvalue(), mimetype='image/png')

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app
