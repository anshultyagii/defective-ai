from sklearn.model_selection import GroupKFold
def get_cv():
    return GroupKFold(n_splits=2)
