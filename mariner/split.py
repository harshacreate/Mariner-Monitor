import numpy as np
from sklearn.model_selection import GroupShuffleSplit
def split_indices(y,groups,seed=42,test_size=.25):
    y=np.asarray(y);groups=np.asarray(groups);idx=np.arange(len(y))
    if y.ndim!=1 or groups.shape!=y.shape or len(np.unique(groups))<2:raise ValueError("invalid target or vessel groups")
    return next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(idx,y,groups))
