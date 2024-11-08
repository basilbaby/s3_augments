import os

class Config:
    DEBUG = True
    SECRET_KEY = os.urandom(24)
