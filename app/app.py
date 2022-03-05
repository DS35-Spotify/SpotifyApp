from os import getenv
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, url_for, redirect
# from .training import *
from .data import *


def create_app():

    # Create a Flask app object
    app = Flask(__name__)

    # Connect our application to the MongoDB Instance
    app.config["MONGO_URI"] = getenv('DATABASE_URI')
    MONGO_DB_CLIENT = PyMongo()
    MONGO_DB_CLIENT.init_app(app)
    DB = MONGO_DB_CLIENT.db

    # HOME ROUTE ************************************************************************

    @app.route("/", methods=['GET', 'POST'])
    def home_page(track_search=None,
                  artist_search=None,
                  search_results=None,
                  preferences=None,
                  recommendations=None):

        # Form Requests
        if request.method == 'POST':

            # Search Form
            if 'track_search' in request.form or 'artist_search' in request.form:
                track_search = request.values['track_search']
                artist_search = request.values['artist_search']
                search_results = search_tracks(artist=artist_search,
                                               name=track_search,
                                               n_tracks=100)

            # Preference Form
            if 'track_preference' in request.form:
                add_tracks_to_db(database=DB,
                                 table_name='tracks',
                                 list_of_track_ids=[
                                     request.values['track_preference']],
                                 preference=1)

            # Recommendation Form
            if 'recommend' in request.form:
                recommendations = n_nearest_tracks(database=DB)
            else:
                recommendations = None

        preferences = [pref for pref in DB['tracks'].find({'preference': 1})]

        return render_template('base.html',
                               search_results=search_results,
                               preferences=preferences,
                               recommendations=recommendations)
    # ************************************************************************************

    # RESET PREFERENCES ******************************************************************
    @app.route('/reset_preferences')
    def reset_preferences():
        '''Resets track preferences'''

        DB['tracks'].delete_many({'preference': 1})

        return redirect(
            url_for('home_page')
        )
    # *************************************************************************************

    # POPULATE ****************************************************************************
    @app.route('/populate')
    def populate(num_tracks=1000):
        '''Populates Tracks in Database'''

        update_suggestion_pool(database=DB, num_tracks=num_tracks)

        return redirect(
            url_for('home_page')
        )
    # *************************************************************************************

    return app
