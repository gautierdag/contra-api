import pickle
from sklearn.linear_model import LogisticRegression
from utils import *

if __name__ == "__main__":
    print("Loading Dataset")
    dataset = get_dataset()
    print("Training Model")
    clf = LogisticRegression(
        random_state=0, multi_class="ovr", solver="lbfgs", max_iter=1000
    )
    clf.fit(dataset[feat_cols], dataset["entailment"])
    print("Saving Model")
    pickle.dump(clf, open("models/log_reg_model.pkl", "wb"))
