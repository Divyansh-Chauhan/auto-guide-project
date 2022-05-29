from flask import Flask, render_template


app = Flask(__name__)

from segment import segment 
from index import index
from location import location
from specification import specification
from timing import timing 
# from test import test

app.register_blueprint(segment)
app.register_blueprint(index)
app.register_blueprint(location)
app.register_blueprint(specification)
app.register_blueprint(timing)

# app.register_blueprint(test)

if __name__=="__main__":
    app.run(debug=True)
    