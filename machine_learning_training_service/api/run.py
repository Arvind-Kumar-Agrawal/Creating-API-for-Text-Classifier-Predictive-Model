#!flask/bin/python
from app import app
import sys
from flask_cors import CORS

cors = CORS(app)
app.run(debug=True, host='localhost', port=9060)
