from flask import Flask, render_template
import os

app = Flask(__name__)

import imeall.views

#@app.errorhandler(404)
#def not_found(error):
#	return render_template('404.html'), 404


