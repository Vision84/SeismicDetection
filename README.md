# SeismicDetection
Implementation for NASA Space Apps Challenge 2024 Seismic Detection Across the Solar System
<br/><br/>
Link to our colab notebook: https://colab.research.google.com/drive/1YL0aAwL80CsV0mz1SVSIsPZNvo1OA2zM?usp=sharing
### Results
The results were added to catalogs: **lunar-s12gradeB.csv, lunar-s15gradeA.csv, lunar-s15gradeB.csv, lunar-s16gradeA.csv, results-mars.csv.**

# Approach
Seismic events are usually characterized by a sudden increase in velocity. To detect such events, we employ a recursive segmentation algorithm that iteratively divides the time-series data into smaller segments, identifies the intervals where a seismic event is likely to occur, and further narrows down the segments to precisely locate the event start time.

### Initial Segmentation:
The algorithm starts by dividing the data into 20,000-second intervals. Each interval is analyzed for seismic event likelihood based on predictions from the deep learning model.
### Recursive Refinement: 
If an event is detected within a segment, the segment is recursively subdivided into smaller segments (dividing by 2), and the analysis is repeated on each smaller segment to find the exact starting point of the seismic event.

# Deep Learning
A key component of the approach is the use of a deep learning model (Convolutional Neural Network) that is trained to classify each time segment as either containing a seismic event or not. The model processes the plotted segments of velocity data and predicts whether an event is present.
Each time segment is plotted as an image, which is then passed through the model. The model classifies the image, and if an event is detected, the recursive segmentation continues on that specific time segment. This method allows us to visualize the data and leverage powerful image-based models for time-series classification.
The plotted time-series images are preprocessed by converting them into tensors, resizing them to a fixed size (300x1000), and passing them through a neural network. The model outputs a prediction, and if the prediction indicates the presence of a seismic event, the recursive search continues.

# Implementation Details:
Python as the programming language.
NumPy for handling numerical data operations.
Pandas for loading and manipulating CSV data.
Matplotlib for plotting time-series data into images.
PyTorch for the neural network model.
Flask for the interface backend.
ngrok for exposing the local Flask server to the internet, allowing users to upload their CSV files for analysis.

# Web Interface and Deployment:
The web app allows users to upload a CSV file containing time-series velocity data. Once uploaded, the data is processed by the backend, and the detected seismic events are returned in the form of time intervals. As we are using a large deep learning model, we couldn't deploy our website for now, that's we we're using ngrok to expose the local flask server to the internet
