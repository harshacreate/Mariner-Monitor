import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,confusion_matrix,precision_score,recall_score,roc_auc_score

class RobustZScore:
    def fit(self,X):
        X=np.asarray(X,dtype=float);self.center=np.median(X,axis=0);q1,q3=np.quantile(X,[.25,.75],axis=0);self.scale=np.maximum(q3-q1,1e-6);return self
    def score_samples(self,X):return np.max(np.abs((np.asarray(X)-self.center)/self.scale),axis=1)

class IsolationForestScore:
    def __init__(self,seed=42):self.model=IsolationForest(n_estimators=200,max_samples="auto",contamination="auto",random_state=seed,n_jobs=1)
    def fit(self,X):self.model.fit(X);return self
    def score_samples(self,X):return -self.model.decision_function(X)

def fit_models(X_normal,seed=42):
    return {"robust_zscore":RobustZScore().fit(X_normal),"isolation_forest":IsolationForestScore(seed).fit(X_normal)}

def threshold_at_false_alarm(model,X_normal,quantile=.95):return float(np.quantile(model.score_samples(X_normal),quantile))

def evaluate(model,X,y,threshold):
    y=np.asarray(y);score=model.score_samples(X);pred=(score>=threshold).astype(int);cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    return {"n":int(len(y)),"roc_auc":round(float(roc_auc_score(y,score)),4),"average_precision":round(float(average_precision_score(y,score)),4),"threshold":round(float(threshold),6),"precision":round(float(precision_score(y,pred,zero_division=0)),4),"recall":round(float(recall_score(y,pred,zero_division=0)),4),"confusion_matrix":cm,"false_positives":int(cm[0][1]),"false_negatives":int(cm[1][0])}
