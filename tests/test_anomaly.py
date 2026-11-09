import json,os,subprocess,sys
import numpy as np
from mariner.data import FEATURES,make_dataset
from mariner.split import split_indices
from mariner.train import run
def test_data_and_group_split():
    X,y,g=make_dataset(600,2);X2,y2,g2=make_dataset(600,2)
    assert X.shape==(600,len(FEATURES)) and np.isfinite(X).all() and np.array_equal(X,X2) and np.array_equal(y,y2)
    tr,te=split_indices(y,g,4);assert not set(g[tr]).intersection(g[te])
def test_unsupervised_scores_and_cli():
    r=run(9,600)
    for key in ("robust_zscore_baseline","isolation_forest"):
        assert 0<=r[key]["average_precision"]<=1 and r[key]["false_positives"]>=0
    env=dict(os.environ);env["PYTHONPATH"]=os.pathsep.join([os.getcwd(),env.get("PYTHONPATH","")])
    p=subprocess.run([sys.executable,"-m","mariner.train","--samples","600","--json"],check=True,capture_output=True,text=True,env=env)
    assert json.loads(p.stdout)["project"]=="Mariner-Monitor — Marine Vessel Anomaly Detection"
