
import os

from flask import Flask

from flask_sqlalchemy import SQLAlchemy
from importlib import import_module
from flask_socketio import SocketIO

# Create SocketIO instance
socketio = SocketIO(cors_allowed_origins="*", logger=True, engineio_logger=True)


db = SQLAlchemy()






def register_blueprints(app):
    for module_name in ('home',):  # Note the comma after 'home'

        module = import_module('apps.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)




    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    
    register_blueprints(app)
    
    socketio.init_app(app)  # Initialize SocketIO here
    return app
