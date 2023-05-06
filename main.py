import requests
import numpy as np
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from flask import jsonify, Flask
from flask_cors import CORS, cross_origin
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd

app = Flask(__name__)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, origins="*")

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

#Setup Spotify and Firebase Credentials
sp = None
def spotify_client():
    global sp
    sp_cred = None
    with open('spotify_credentials.json') as credentials:
        sp_cred = json.load(credentials)
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(sp_cred["client_id"],sp_cred['client_secret']))


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
        return (jsonify({"error":"Bad Input, must pass user_id, map of songs and their metadata, and mood label"}), 
                400)
    if cred == None:
        firestoreConnection()
    res = retrieveCentroid(user_id, mood)
    if "error" in res:
        return (jsonify(res), 400)
    centroid = res["centroid"]
    song_scores_dict = getSongScores(songs)
    if song_scores_dict is None: return (jsonify({"error": "could not find spotify features for song ids"}), 400)
    print(song_scores_dict)
    distances = []
    
    for (name, score) in song_scores_dict.items():
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
    if centroid_dict is None:
        return {"error": f"no centroid for mood - {mood} - found for user with ID {user_id}"}
    sorted_dict = sorted(centroid_dict.items(), key=lambda x: x[0])
    centroid = [v[1] for v in sorted_dict]
    return {"centroid": centroid}

def getSongScores(songs):
    global sp
    if sp == None:
        spotify_client()
    print(f"songs: {songs}")
    track_info = sp.tracks(songs)['tracks']
    # Remove any elements that are None
    track_info = [track for track in track_info if track is not None]
    track_ids = [track["id"] for track in track_info]
    return retrieveTrackFeatures(track_ids)

def retrieveTrackFeatures(track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        features = sp.audio_features(track_ids[i:i+50])
        checked_features = [l for l in features if l is not None]
        if len(checked_features) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(checked_features)
            
            # Remove columns that we don't need
            df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)
            
            # df = df[['id', 'valence', 'energy']]
            
            # Append to list of dataframes
            dfs.append(df)
    if len(dfs) == 0: return None
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    features_dict = features_df.set_index('id').T.to_dict('list')
    return features_dict
    # return features_df

if __name__ == '__main__':
    app = Flask(__name__)
    app.route('/closestSongs', methods=['POST'])(lambda request: closestSongs(request))
    app.run(debug=True)