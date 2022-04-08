
# Imports ********************************************
# For Accessing Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# For Accessing Environment Variables
from os import getenv

# For Handling NumPy Arrays
import numpy as np
from bson.binary import Binary
import pickle

# For Neural Network Recommendations
from tensorflow.keras.models import load_model
from numpy.random import shuffle

# ****************************************************

# SpotiPy Client ******************************************************************
CLIENT_ID = getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = getenv('SPOTIFY_CLIENT_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID,
                                                      client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
# *********************************************************************************

# Neural Network Track Recommender ******************************************************


def neural_net_tracks(database, n_tracks=10):

    # First we need to load the data from the database *******************************

    # Load Playlist
    playlist_vector_list = [unpickle_my_array(track['vector'])
                            for track in database['tracks'].find({'preference': 1})]

    if len(playlist_vector_list) < 10:
        print('ERROR: NOT ENOUGH TRACKS IN PLAYLIST')
        return

    for i, vector in enumerate(playlist_vector_list):
        if i == 0:
            playlist_matrix = playlist_vector_list[i]
        elif i < 10:
            playlist_matrix = np.vstack(
                (playlist_matrix, playlist_vector_list[i]))

    playlist_matrix = playlist_matrix.reshape(1, 10, 13)

    # Load Other Tracks and their Names + Artists
    track_info_list = [(unpickle_my_array(track['vector']), track['name'], track['artists'])
                       for track in database['tracks'].find({'preference': 0})]

    # Format Playlists for Input
    for i in range(len(track_info_list)):
        if i == 0:
            playlists_matrix = playlist_matrix
        else:
            playlists_matrix = np.vstack((playlists_matrix, playlist_matrix))

    # Format Tracks for Input
    for i, info in enumerate(track_info_list):
        if i == 0:
            track_vectors = info[0]
        else:
            track_vectors = np.vstack((track_vectors, info[0]))

    # # Standardize the AUDIO FEATURE Scaling
    # playlists_matrix = (playlists_matrix - playlists_matrix.min(axis=1, keepdims=True)) / \
    #     playlists_matrix.ptp(axis=1, keepdims=True)

    # # SHAPE = (SAMPLES, TRACKS, FEATURES) --> NEED TO NORMALIZE FEATURES
    # print(playlists_matrix.shape)
    # temp = []
    # for a in range(playlist_matrix.shape[0]):
    #     for b in range(playlist_matrix.shape[1]):
    #         for c in range(playlist_matrix.shape[2]):
    #             temp += [playlists_matrix[a][b][0]]
    # print(min(temp), max(temp))

    # track_vectors = (track_vectors - track_vectors.min(axis=0, keepdims=True)) / \
    #     track_vectors.ptp(axis=0, keepdims=True)

    # Now we need to load the model **************************************************

    model = load_model('models/model0')
    targets = model.predict(
        x=[playlists_matrix, track_vectors],
    )

    # Names and Artists
    naa = []

    for i, target in enumerate(targets):
        if target == 1:
            naa.append((track_info_list[i][1], track_info_list[i][2]))

    shuffle(naa)

    if len(naa) <= n_tracks:
        return naa
    else:
        return naa[:n_tracks]


# ***************************************************************************************

# User Song Lookup ****************************************************************


def search_tracks(n_tracks=100, artist=None, name=None):
    '''Returns a list of dictionaries
       Each dictionary contains a track's
       id, name, artists, and album
    '''
    tracks = []

    # track limit
    n_tracks = min(n_tracks, 1000)

    # generate query string
    query = ''
    if artist:
        query += f'artist:{artist}'
        if name:
            query += f' track:{name}'
    else:
        if name:
            query += f'track:{name}'
        else:
            print('error: no search parameters entered')
            return

    # results are limited to 1000 items
    #  and each search will only return 50 items
    #  so we have to loop over them with an offset index
    n = 0  # number of tracks found
    for i in range(0, 1000, 50):
        result = sp.search(q=query,
                           type='track',
                           limit=50,
                           offset=i)['tracks']['items']
        for track in result:
            tracks.append({'id': track['id'],
                           'name': track['name'],
                           'artists': track['artists'][0]['name'],
                           'album': track['album']['name']})
            n += 1
            # stops looking at songs if enough were found
            if n >= n_tracks:
                break
        # stops querying if spotify is out of results
        # or if enough tracks were found
        if len(result) < 50 or n >= n_tracks:
            break
    return tracks
# ********************************************************************************


# Helper Functions ******************************************************


def pickle_my_array(array):
    return Binary(pickle.dumps(array, protocol=2), subtype=128)


def unpickle_my_array(pickled_array):
    return pickle.loads(pickled_array)


def list_collections_names(database):
    print(database.list_collection_names())

# ************************************************************************


# Database Management *************************************************************


def add_tracks_to_db(database, table_name, list_of_track_ids, preference=0):

    list_of_track_ids = list(set(list_of_track_ids))

    # Search for Names and Artists (50 at a time)
    track_info = []
    offset = 0
    while offset < len(list_of_track_ids):
        last_index = offset + min(50, len(list_of_track_ids)-offset)
        track_info += sp.tracks(list_of_track_ids[offset:last_index])['tracks']
        offset += 50

    # Search for Audio Features (100 at a time)
    vector_info = []
    offset = 0
    while offset < len(list_of_track_ids):
        last_index = offset + min(100, len(list_of_track_ids)-offset)
        vector_info += sp.audio_features(list_of_track_ids[offset:last_index])
        offset += 100

    # Create list of dictionaries to add to MongoDB
    tracks = []
    for i, track_id in enumerate(list_of_track_ids):
        # Delete the track if it is already in the database
        same_ids = [_ for _ in database['tracks'].find({'_id': track_id})]
        if (len(same_ids) > 0):
            database['tracks'].delete_one({'_id': track_id})

        # SOMETIMES SPOTIFY WON'T HAVE TRACK INFO
        # THESE TRACKS WILL BE SKIPPED
        if vector_info[i] and track_info[i]:
            array = np.array([
                vector_info[i]['acousticness'],
                vector_info[i]['danceability'],
                vector_info[i]['duration_ms'],
                vector_info[i]['energy'],
                vector_info[i]['instrumentalness'],
                vector_info[i]['key'],
                vector_info[i]['liveness'],
                vector_info[i]['loudness'],
                vector_info[i]['mode'],
                vector_info[i]['speechiness'],
                vector_info[i]['tempo'],
                vector_info[i]['time_signature'],
                vector_info[i]['valence']])
            tracks.append({'_id': track_id,
                           'name': track_info[i]['name'],
                           'artists': track_info[i]['artists'][0]['name'],
                           'preference': preference,
                           'vector': pickle_my_array(array)
                           })
        else:
            print(
                f'Error: No Audio Features Found For {track_id} - {track_info[i]["name"]}')

    database[table_name].insert_many(tracks)

    return


def update_suggestion_pool(database, num_tracks=1000):

    # Request Cap
    num_tracks = min(num_tracks, 1000)

    # Parameters
    max_tracks_per_album = 3

    # Get Prior Track Ids (Prevent Duplicates)
    stored_ids = [thing['_id'] for thing in database['tracks'].find({})]

    # Get Track Ids
    track_ids = []
    album_offset = 0
    while len(track_ids) < num_tracks:

        # Find NEW, HIP Album IDs
        album_ids = [album['id'] for album in sp.search(q="tag:hipster tag:new",
                                                        type='album',
                                                        market='US',
                                                        limit=50,
                                                        offset=album_offset)['albums']['items']
                     ]

        # Find Tracks Ids
        for album_id in album_ids:
            tracks = sp.album_tracks(album_id,
                                     limit=max_tracks_per_album)['items']
            cutoff = min(num_tracks - len(track_ids), max_tracks_per_album)
            new_ids = [track['id']
                       for track in tracks if (track['id'] not in stored_ids)][0:cutoff]
            track_ids += new_ids
            stored_ids += new_ids

        # Update offset index for next set of albums
        album_offset += 50

    add_tracks_to_db(database=database, table_name='tracks',
                     list_of_track_ids=track_ids, preference=0)

    return
# *******************************************************************************
