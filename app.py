from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np

from preprocess import preprocess_pair

app = Flask(__name__)
api = Api(app)

# load model
clf_path = "models/svm_classifier.pkl"
with open(clf_path, "rb") as f:
    model = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument("text")
parser.add_argument("hyp")


class PredictContradiction(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args["query"]

        # vectorize the user's query and make a prediction
        X = preprocess_pair(t, h)
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = "Negative"
        else:
            pred_text = "Positive"

        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {"prediction": pred_text, "confidence": confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictContradiction, "/")


if __name__ == "__main__":
    app.run(debug=True)
