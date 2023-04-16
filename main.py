import functions_framework
import requests
import numpy as np
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

#may need to be used if mood_index is passed instead of the mood itself
moods = ['sad','angry','energetic','excited','happy','content','calm','depressed'] 

headers = { 
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
}   

#Initialize credentials to database
cred = None
def firestoreConnection(): 
    global cred
    cred = credentials.Certificate("mood-swing-6c9d0-firebase-adminsdk-9cm02-66f39cc0dd.json")
    firebase_admin.initialize_app(cred)

#Parameters: user, dict of song names and their list of metadata values, the mood for the centroid to retrieve
@functions_framework.http
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
        return ({"error":"Bad Input, must pass user_id, map of songs and their metadata, \
                 and mood label"}, 400, headers)
    if cred == None:
        firestoreConnection()
    centroid = retrieveCentroid(user_id, mood)
    if len(songs) <= 5: return (songs.keys(), 200, headers)
    distances = []
    for (name, score) in songs.items():
        calculated_distance = cosineSimilarity(centroid, score)
        distances.append((name,calculated_distance))
    #sort the distances by value
    distances = sorted(distances, key=lambda x: x[1])
    print(distances[:5])
    #return the song names of the 5 smallest distances
    closest_songs = [pair[0] for pair in distances[:5]]
    return (closest_songs, 200, headers)

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