import os
import sys
import logging
from   logging.handlers import RotatingFileHandler

try:
    from flask import Flask, render_template
    NO_FLASK = False
except:
    NO_FLASK = True
    print 'No Flask Server Available'

def file_extension(filepath):
    endswith = filepath.split('.')[-1]
    return endswith

if not NO_FLASK:
    class PrefixMiddleware(object):
        def __init__(self, app, prefix=''):
            self.app = app
            self.prefix = prefix

        def __call__(self, environ, start_response):
            if environ['PATH_INFO'].startswith(self.prefix):
                environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
                environ['SCRIPT_NAME'] = self.prefix
                return self.app(environ, start_response)
            else:
                start_response('404', [('Content-Type', 'text/plain')])
                return ["This url does not belong to the app.".encode()]

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile('config.py')
    app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix=app.config['APPLICATION_ROOT'])
    handler = RotatingFileHandler('imeall.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.jinja_env.filters['file_extension'] = file_extension

    from imeall import views
