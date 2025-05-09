from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier

def predict_svc_self(X, y):
    base_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model = SelfTrainingClassifier(base_model, threshold=0.95)

    model.fit(X, y)

    predictions = model.predict(X)
    return predictions