import pickle
import os
from sklearn.linear_model import LogisticRegression
from utils import *

MODEL_PATH = "models/log_reg_model.pkl"


def get_model(model_path=MODEL_PATH):
    """
    Returns model at model_path
    Args: 
        model_path (path, optional): path to model
    Returns:
        clf (sklearn Model): sk learn model instance trained
                             or model contained at path
    """

    # Check if path exists by loading
    try:
        clf = pickle.load(open(model_path, "rb"))

    # If fails then build model
    except (OSError, IOError) as e:
        # No model found - build model
        print("Loading Dataset")
        dataset = get_dataset()
        print("Training Model")
        clf = LogisticRegression(
            random_state=0, multi_class="ovr", solver="lbfgs", max_iter=1000
        )
        clf.fit(dataset[feat_cols], dataset["entailment"])
        print("Saving Model")
        pickle.dump(clf, open(model_path, "wb"))

    return clf


if __name__ == "__main__":
    get_model()
