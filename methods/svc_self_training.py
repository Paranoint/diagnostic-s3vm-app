import numpy as np
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier

def predict_svc_self(X_labeled, y_labeled, X_unlabeled):
    base_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model = SelfTrainingClassifier(base_model, threshold=0.95)

    X = np.concatenate([X_labeled, X_unlabeled])
    y = np.concatenate([y_labeled, [-1] * len(X_unlabeled)])

    model.fit(X, y)

    return model.predict(X)