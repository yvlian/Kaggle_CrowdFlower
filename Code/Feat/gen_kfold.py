
"""
__file__

    gen_kfold.py

__description__

    This file generates the StratifiedKFold indices which will be kept fixed in
    ALL the following model building parts.

__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
import _pickle as  cPickle
from sklearn.model_selection import StratifiedKFold
sys.path.append("../")
from param_config import config


if __name__ == "__main__":

    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)

    skf_data = [0] * config.n_runs
    for key,stratified_label in zip(["relevance_variance", "query"], ["median_relevance", "qid"]):
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run+1)
            skf = StratifiedKFold(n_splits=config.n_folds,
                                  shuffle=True, random_state=random_seed)
            skf_data[run] = [(fold, (validInd, trainInd)) for fold, (validInd, trainInd)
                             in enumerate(skf.split(dfTrain[key],dfTrain[stratified_label]))]
        with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, key), "wb") as f:
            cPickle.dump(skf_data, f, -1)