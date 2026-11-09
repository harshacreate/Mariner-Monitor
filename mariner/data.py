"""Synthetic vessel telemetry; anomalies are injected for offline evaluation."""
from __future__ import annotations
import numpy as np
FEATURES=["speed_knots","acceleration","course_change_deg","navigation_error","message_gap_seconds","wind_speed"]
def make_dataset(n=3000,seed=42):
    if n<200:raise ValueError("n must be at least 200")
    rng=np.random.default_rng(seed);vessels=max(20,n//50);groups=np.arange(n)%vessels
    y=(rng.random(n)<.10).astype(int);speed=rng.normal(12,3,n);accel=rng.normal(0,.35,n);course=np.abs(rng.normal(3,3,n));nav=np.abs(rng.normal(.5,.25,n));gap=np.clip(rng.lognormal(np.log(60),.25,n),15,180);wind=np.clip(rng.normal(15,6,n),0,45)
    kind=rng.integers(0,4,n);speed+=y*(5+12*(kind==0));accel+=y*rng.normal(0,1.8,n)*(kind==1);course+=y*rng.uniform(35,150,n)*(kind==2);nav+=y*rng.uniform(2,10,n)*(kind==3);gap=np.where((y==1)&(kind==0),gap*3,gap)
    X=np.column_stack([speed,accel,course,nav,gap,wind]).astype(float)
    if X.shape!=(n,len(FEATURES)) or not np.isfinite(X).all():raise ValueError("invalid telemetry")
    return X,y,groups
