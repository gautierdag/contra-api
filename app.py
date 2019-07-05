from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np

from build_model import get_model
from preprocess import preprocess_pair

app = Flask(__name__)
api = Api(app)

# Get model
clf = get_model()

# Confidence dictionary to map label to probability
confidence_dict = {clf.classes_[i]: i for i in range(len(clf.classes_))}

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("txt")
parser.add_argument("hyp")


class PredictContradiction(Resource):
    def get(self):
        # use parser and find the queried "text" and "hyp"
        args = parser.parse_args()
        t = args["txt"]
        h = args["hyp"]

        # vectorize the user's query and make a prediction
        X = np.array(preprocess_pair(t, h)).reshape(1, -1)
        prediction = clf.predict(X)
        pred_proba = clf.predict_proba(X)

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0][confidence_dict[prediction[0]]], 3)
        # create JSON object
        output = {"prediction": prediction[0], "confidence": confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictContradiction, "/")


if __name__ == "__main__":
    app.run()
