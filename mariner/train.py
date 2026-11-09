import argparse,json
from .data import make_dataset
from .split import split_indices
from .model import fit_models,evaluate,threshold_at_false_alarm
PROJECT="Mariner-Monitor — Marine Vessel Anomaly Detection"
def run(seed=42,samples=3000):
    X,y,groups=make_dataset(samples,seed);tr,te=split_indices(y,groups,seed);normal_train=X[tr][y[tr]==0]
    models=fit_models(normal_train,seed);report={"project":PROJECT,"data":"synthetic telemetry with injected anomalies","seed":int(seed),"split":"vessel-group holdout; fit on normal training rows only","n_train":int(len(tr)),"n_test":int(len(te)),"normal_training_rows":int(len(normal_train)),"threshold_policy":"95th percentile of each model's training-normal anomaly scores (nominal 5% training false-alarm threshold)"}
    report["robust_zscore_baseline"]=evaluate(models["robust_zscore"],X[te],y[te],threshold_at_false_alarm(models["robust_zscore"],normal_train))
    report["isolation_forest"]=evaluate(models["isolation_forest"],X[te],y[te],threshold_at_false_alarm(models["isolation_forest"],normal_train))
    return report
def main(argv=None):
    p=argparse.ArgumentParser(description="Evaluate unsupervised synthetic vessel anomaly scorers.");p.add_argument("--seed",type=int,default=42);p.add_argument("--samples",type=int,default=3000);p.add_argument("--json",action="store_true");a=p.parse_args(argv);print(json.dumps(run(a.seed,a.samples),indent=2))
if __name__=="__main__":main()
