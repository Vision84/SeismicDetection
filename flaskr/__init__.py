import os
import json

from flask import (Flask, render_template, redirect, request, url_for, g, Blueprint)
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
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        # FILE IS SAVED TO ./uploads/ - PROCESS USING MODEL HERE AND RETURN OUTPUT

        return json.dumps({"output": 2.0})
    

    return app