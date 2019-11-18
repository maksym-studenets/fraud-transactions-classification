from imblearn.over_sampling import SMOTE

def resampleWithSmote(xTrain, yTrain):
    smote = SMOTE(random_state=2)
    xTrainRes, yTrainRes = smote.fit_sample(xTrain, yTrain.ravel())
