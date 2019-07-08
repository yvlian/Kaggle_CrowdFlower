from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNC
from hyperopt import tpe,Trials,hp,fmin,STATUS_OK

data = load_iris()
X = data['data']
y = data['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def hyperopt_train_test(params):
    clf = KNC(**params)
    return cross_val_score(clf, X, y).mean()


space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,100))
}


def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=1000, trials=trials)
print('best:',best)