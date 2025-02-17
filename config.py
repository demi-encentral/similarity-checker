# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or 'your-secret-key'
    MODEL_ID = os.environ.get("MODEL_ID")
    FORM_URL = os.environ.get("FORM_URL")