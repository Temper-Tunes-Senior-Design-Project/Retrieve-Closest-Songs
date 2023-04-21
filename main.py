import requests
import numpy as np
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import jsonify, Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

#Initialize credentials to database
cred = None
def firestoreConnection(): 
    global cred
    cred = credentials.Certificate("mood-swing-6c9d0-firebase-adminsdk-9cm02-66f39cc0dd.json")
    firebase_admin.initialize_app(cred)

#Parameters: user, dict of song names and their list of metadata values, the mood for the centroid to retrieve
@app.route('/closestSongs')
@cross_origin()
def closestSongs(request):
    request_json = request.get_json(silent=True)
    user_id = '' 
    songs = {}
    mood = ''
    if request_json and all(k in request_json for k in ("user_id","songs","mood")):
        user_id = request_json["user_id"]
        songs = request_json["songs"]
        mood = request_json["mood"]
    else:
        return ({"error":"Bad Input, must pass user_id, map of songs and their metadata, and mood label"}, 
                400)
    if cred == None:
        firestoreConnection()
    centroid = retrieveCentroid(user_id, mood)
    distances = []
    for (name, score) in songs.items():
        calculated_distance = cosineSimilarity(centroid, score)
        distances.append((name,calculated_distance))
    #sort the distances by value
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    #return the song names in order of closest to the centroid
    closest_songs = [pair[0] for pair in distances]
    return (jsonify({'songs': closest_songs}), 200)

def cosineSimilarity(arr1, arr2):
    return np.dot(arr1, arr2)/(norm(arr1)*norm(arr2))
        
def retrieveCentroid(user_id, mood):
    # Get a reference to the users collection
    users_ref = firestore.client().collection("users")
    # Get a reference to the specific user's document
    user_doc_ref = users_ref.document(user_id)
    # Get a reference to the "mood" document in the centroids subcollection
    mood_doc_ref = user_doc_ref.collection("centroids").document(mood) 
    # Get the centroid in dict format, and sort the centroid keys
    centroid_dict = mood_doc_ref.get().to_dict()
    sorted_dict = sorted(centroid_dict.items(), key=lambda x: x[0])
    centroid = [v[1] for v in sorted_dict]
    return centroid

if __name__ == '__main__':
    app = Flask(__name__)
    app.route('/closestSongs', methods=['POST'])(lambda request: closestSongs(request))
    app.run(debug=True)