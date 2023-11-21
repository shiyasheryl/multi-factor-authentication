import base64

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import cv2
import dlib
import numpy as np
import argparse
import imutils
import pymongo
from bson.binary import Binary
from imutils.face_utils import visualize_facial_landmarks

app = Flask(__name__)
CORS(app, resources={r"/capture": {"origins": "http://localhost:5173", r"/match": {"origins": "http://localhost:5173"}}}) # Replace with the correct origin of your React app.


CORS(app, supports_credentials=True, origins='http://localhost:5173')

# Define the indexes for facial landmarks.
facial_features_cordinates={}
FACIAL_LANDMARKS_INDEXES = {
    "Mouth": (48, 68),
    "Right_Eyebrow": (17, 22),
    "Left_Eyebrow": (22, 27),
    "Right_Eye": (36, 42),
    "Left_Eye": (42, 48),
    "Nose": (27, 35),
    "Jaw": (0, 17),
}

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sheryldekshanya/PycharmProjects/fc/Detect-Facial-Features/shape_predictor_68_face_landmarks.dat")  # Replace with the correct path
def euclidean_distance(ref_pts,live_pts):
    distance = np.linalg.norm(np.array(ref_pts) - np.array(live_pts))
    return distance

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
            else:
                hull = cv2.convexHull(pts)
                cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    # print(facial_features_cordinates)
    return output


# Create a route to display the capture form
@app.route('/')
def index():
    return render_template('index.html')

# Create a route to capture and process images
@app.route('/capture', methods=['POST'])
# ... (other code)

# Create a route to capture and process images
def capture():
    data = request.get_json()  # Get JSON data from the request body
    name = data.get('name')  # Get the "name" parameter from the JSON data
    captured_image = data.get('captured_image')

    image_bytes = captured_image.split(',')[1].encode()
    image_array = np.frombuffer(base64.b64decode(image_bytes), np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)


    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        # Visualize facial landmarks on the frame
        frame = visualize_facial_landmarks(frame, shape)
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)
        output = visualize_facial_landmarks(frame, shape)
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["admin"]  # Replace with your database name
        collection = db["face features"]  # Replace with your collection name
        for key, value in facial_features_cordinates.items():
            if isinstance(value, np.ndarray):
                facial_features_cordinates[key] = value.tolist()
        fc = facial_features_cordinates.copy()
        person_name = name
        dataa = {

            "name": person_name,
            "features": fc
        }
        print("features stored", dataa)

        result = collection.insert_one(dataa)

        print("Inserted document ID:", result.inserted_id)
        response = {
            "message": "Image captured and processed successfully.",
            "status": "success"
        }
        client.close()

        # You can use a proper status code (e.g., 201 Created) for success
    return jsonify({'message': 'Image captured and processed successfully.'}), 200

        # Load the target feature set you want to match (e.g., from your application)





@app.route('/match')
def index1():
    return render_template('match.html')

# Route to handle form submissions
# Route to handle form submissions
@app.route('/match', methods=['POST'])
def match():
    response_data = {}
    data = request.get_json()  # Get JSON data from the request body
    name = data.get('name')  # Get the "name" parameter from the JSON data
    captured_image = data.get('captured_image')

    image_bytes = captured_image.split(',')[1].encode()
    image_array = np.frombuffer(base64.b64decode(image_bytes), np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        # Visualize facial landmarks on the frame
        frame = visualize_facial_landmarks(frame, shape)
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)
        output = visualize_facial_landmarks(frame, shape)
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["admin"]  # Replace with your database name
        collection = db["face features"]  # Replace with your collection name
        # Load the target feature set you want to match (e.g., from your application)
        for key, value in facial_features_cordinates.items():
            if isinstance(value, np.ndarray):
                facial_features_cordinates[key] = value.tolist()

        target_features = {

            "features": facial_features_cordinates
        }
        print(target_features)

        document_feature = collection.find_one({"name": name})
        print(document_feature)

        landmark_distances = {}

        # Loop through the keys (facial landmarks)
        for landmark in document_feature['features']:
            if landmark in ["_id", "name"]:
                continue

            # Access the arrays of values within each dictionary
            ref_pts = document_feature['features'][landmark]
            live_pts = target_features['features'][landmark]

            # Perform element-wise subtraction

            # Calculate the Euclidean distance for the resulting array (if needed)
            distance = euclidean_distance(ref_pts, live_pts)

            # Store the result or distance in your landmark_distances dictionary
            landmark_distances[landmark] = distance
            # You can aggregate the distances here, e.g., by taking the mean
            aggregated_distance = np.mean(distance)

            landmark_distances[landmark] = aggregated_distance

        # Define a threshold for similarity (you may need to adjust this)
        print(landmark_distances)
        threshold = 250.0

        # Check if the aggregated distances are below the threshold
        is_similar = all(landmark_distances[landmark] < threshold for landmark in landmark_distances)
        client.close()


        if is_similar:
            print("Faces are similar")
            response_data = {"result": "Faces are similar"}
        else:
            print(landmark_distances)
            print("Faces are dissimilar")
            response_data = {"result": "Faces are dissimilar"}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(port=8080, debug=True)

