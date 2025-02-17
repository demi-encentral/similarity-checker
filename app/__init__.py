from flask import Flask
from config import Config
from logging.handlers import RotatingFileHandler
import logging


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app

