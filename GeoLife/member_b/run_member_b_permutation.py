from __future__ import annotations
import gc, hashlib, json, os, sys, time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
WORK = BASE / "work" / "member_b_permutation"
RESULTS = BASE / "member_b" / "outputs"
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / ".member_b_libs"))
sys.path.insert(0, str(ROOT / "tools"))
from member_b_bootstrap import bootstrap_research
bootstrap_research(ROOT)
import joblib
import numpy as np
import pandas as pd
from research import features
from research.deployment_grid import Windows
from research.deployment_protocol import load_config, iter_conditions, degrade_window, window_quality
from sklearn.metrics import f1_score

GROUPS = {
 "speed": ["mean_speed","std_speed","max_speed","median_speed","q25_speed","q75_speed"],
 "acceleration": ["mean_acc","std_acc","max_acc"],
 "heading_change": ["mean_heading","sum_heading","std_heading"],
 "distance": ["path_length"],
 "sampling_time": ["duration","n_points","mean_gap"],
 "stop_ratio": ["stop_ratio"],
}
NAMES = ["ds0_n0","ds10_n0","ds0_n20","ds10_n20"]
SEED = 20261112
REPEATS = 10
RUN = "04d2f3dee1b64ac6e6d359b0aae75128a001a8cd25ac2f4df9afce4a389a1d50"
CLASSES = ["walk","bike","bus","car","subway"]

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
 return h.hexdigest()

def permutation(n,user,group,repeat):
 token=f"{SEED}|{user}|{group}|{repeat}".encode()
 entropy=int.from_bytes(hashlib.sha256(token).digest()[:8],"big")
 return np.random.default_rng(entropy).permutation(n)

def fold(user,data,config,conditions,inventory):
 target=WORK/f"fold_{user}.npz"
 if target.exists():
  print("reuse",user,flush=True)
  return
 selected=np.flatnonzero(data.meta["user_id"].eq(user).to_numpy() & data.support["common"])
 if len(selected)==0: return
 ids=data.meta.iloc[selected]["window_id"].to_numpy(dtype=int)
 labels=data.meta.iloc[selected]["label"].to_numpy(dtype=str)
 row=inventory.loc[user]
 model_path=ROOT/row["rf_path"]
 meta_path=ROOT/row["models_metadata_path"]
 metadata=json.loads(meta_path.read_text(encoding="utf-8"))
 if metadata["identity"]["test_user"]!=user or user in metadata["identity"]["training_users"]: raise ValueError("Fold identity error")
 if row["run_sha256"]!=RUN or metadata["identity"]["run_identity"]!=RUN: raise ValueError("Run mismatch")
 if digest(model_path)!=row["rf_sha256"] or metadata["rf_sha256"]!=row["rf_sha256"]: raise ValueError("RF hash mismatch")
 model=joblib.load(model_path)
 model.n_jobs=1
 columns=features.CORE_A_FEATURES
 if sorted(sum(GROUPS.values(),[]))!=sorted(columns): raise ValueError("Feature groups mismatch")
 predictions=np.empty((len(NAMES),len(GROUPS),REPEATS,len(selected)),dtype=np.uint8)
 baseline=np.empty((len(NAMES),len(selected)),dtype=np.uint8)
 for ci,name in enumerate(NAMES):
  frames=[]
  for index,wid in zip(selected,ids):
   window=degrade_window(data.frame(int(index)),conditions[name],noise_seed=0,window_id=int(wid))
   if not window_quality(window,config)["usable"]: raise ValueError("Unusable common window")
   frames.append(window)
  X=features.build_core_a_matrix(frames).replace([np.inf,-np.inf],np.nan).fillna(0.0)[columns].to_numpy(dtype=np.float32)
  saved_path=ROOT/f"project/artifacts/member_a/run_{RUN[:16]}/folds/{user}/{name}_seed0.csv.gz"
  saved=pd.read_csv(saved_path,dtype={"user_id":str})
  saved=saved.loc[saved["common"].astype(bool)]
  if saved["window_id"].duplicated().any() or set(saved["window_id"].to_numpy(dtype=int))!=set(ids): raise ValueError("Window identity mismatch")
  saved=saved.set_index("window_id").loc[ids]
  pred=model.predict(X).astype(np.uint8)
  if not np.array_equal(pred,saved["pred_rf"].to_numpy(dtype=np.uint8)): raise ValueError(f"Baseline prediction mismatch {user}/{name}")
  baseline[ci]=pred
  for gi,(group,members) in enumerate(GROUPS.items()):
   loc=[columns.index(col) for col in members]
   for repeat in range(REPEATS):
    order=permutation(len(selected),user,group,repeat)
    changed=X.copy()
    changed[:,loc]=X[order][:,loc]
    predictions[ci,gi,repeat]=model.predict(changed).astype(np.uint8)
  print("fold",user,name,len(selected),flush=True)
  del frames,X
 np.savez_compressed(target,ids=ids,labels=labels,baseline=baseline,predictions=predictions)
 del model
 gc.collect()

def aggregate(users):
 packages=[np.load(WORK/f"fold_{u}.npz") for u in users]
 y=np.concatenate([p["labels"] for p in packages])
 base=np.concatenate([p["baseline"] for p in packages],axis=1)
 perm=np.concatenate([p["predictions"] for p in packages],axis=3)
 if len(y)!=71941: raise ValueError(f"Expected 71941 windows; got {len(y)}")
 labels=np.array(CLASSES)
 detail=[]
 summary=[]
 for ci,name in enumerate(NAMES):
  original=f1_score(y,labels[base[ci]],labels=CLASSES,average="macro",zero_division=0)
  for gi,group in enumerate(GROUPS):
   scores=[]
   for repeat in range(REPEATS):
    changed=f1_score(y,labels[perm[ci,gi,repeat]],labels=CLASSES,average="macro",zero_division=0)
    importance=original-changed
    scores.append(importance)
    detail.append(dict(run_hash=RUN,cohort="common",scope="full_louo",condition=name,noise_seed=0,feature_group=group,repeat=repeat,permutation_seed=SEED,n_windows=len(y),n_users=len(users),original_macro_f1=original,permuted_macro_f1=changed,importance=importance))
   summary.append(dict(run_hash=RUN,cohort="common",scope="full_louo",condition=name,noise_seed=0,feature_group=group,n_repeats=REPEATS,mean_importance=np.mean(scores),sd_importance=np.std(scores,ddof=1),n_windows=len(y),n_users=len(users),original_macro_f1=original))
 pd.DataFrame(detail).to_csv(RESULTS/"group_permutation_importance.csv",index=False)
 pd.DataFrame(summary).to_csv(RESULTS/"group_permutation_summary.csv",index=False)
 record=dict(status="complete",run_hash=RUN,cohort="common",noise_seed=0,permutation_repeats=REPEATS,permutation_seed=SEED,participating_users=users,feature_groups_version="v1.0.0",feature_groups=GROUPS,n_windows=len(y),n_users=len(users),rf_n_jobs=1,permutation_rule="same within-user row permutation for all columns in a group, reused across four conditions",source_package_manifest_sha256=(digest(ROOT/"PACKAGE_MANIFEST.json") if (ROOT/"PACKAGE_MANIFEST.json").exists() else None),fold_rf_sha256={u:inventory.loc[u,"rf_sha256"] for u in users})
 (RESULTS/"member_b_permutation_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
 for p in packages:p.close()
 print("aggregate complete",len(detail),len(summary),flush=True)

if __name__=="__main__":
 cfg=load_config(ROOT/"现实关联"/"degradation_config.yaml")
 data=Windows(cfg)
 conditions={c.name:c for c in iter_conditions(cfg) if c.name in NAMES}
 inventory=pd.read_csv(ROOT/"现实关联"/"outputs"/"member_a_fold_inventory.csv",dtype={"test_user":str}).set_index("test_user")
 users=sorted(data.meta.loc[data.support["common"],"user_id"].unique())
 if len(users)!=48: raise ValueError("Common user count mismatch")
 for user in users:fold(user,data,cfg,conditions,inventory)
 aggregate(users)
