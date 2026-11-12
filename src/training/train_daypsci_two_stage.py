from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1")
VALIDATION_BATCH = "pn_sensor_T1_batch02"
TEST_BATCH = "pn_sensor_T2_batch03"
BASELINE_BATCH = "pn_baseline__batch01"
BATCHES = [
    "pn_baseline__batch01", "pn_combined_T1_batch11", "pn_sensor_T1_batch02",
    "pn_sensor_T2_batch03", "pn_sensor_T3_batch04", "pn_sensor_T4_batch05",
    "pn_sensor_T5_batch06", "pn_sensor_T6_batch07", "pn_valve_T1_batch08",
    "pn_valve_T2_batch09", "pn_valve_T3_batch10",
]
NAMES = {0:"Normal", 1:"Sensor anomaly", 2:"Valve anomaly", 3:"Combined anomaly", 4:"Propagated anomaly"}
EXCLUDED = {"anomaly","fault","FSx","TFSx","FCVx","TFCVx","batch_id","experiment_id","part_id","event_seq","ts_plc","event_id"}
SIGNALS = ["a0","a1","b0","b1","c0","c1","B_1","B_2","YA_p","YA_m","YB_p","YB_m","YC_p","YC_m"]

def find_csv(batch):
    folder = DATASET_DIR / batch
    if not folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {folder}")
    files = list(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {folder}")
    matches = [p for p in files if batch in p.name]
    return matches[0] if matches else files[0]

def load_batch(batch):
    df = pd.read_csv(find_csv(batch), sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "anomaly" not in df.columns:
        raise ValueError(f"Missing anomaly column in batch {batch}")
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce")
    df = df[df["anomaly"].isin([0,1,2,3,4])].copy()
    df["anomaly"] = df["anomaly"].astype(int)
    df["_batch"] = batch
    sort_cols = [c for c in ["part_id","event_seq","scan_id","event_order"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    group = df["part_id"] if "part_id" in df.columns else df["_batch"]
    if "delta_time_ms" in df.columns:
        dt = pd.to_numeric(df["delta_time_ms"], errors="coerce")
        df["delta_prev_ms"] = dt.groupby(group).shift(1)
        df["delta_change_ms"] = dt - df["delta_prev_ms"]
        df["delta_rolling_mean_3"] = dt.groupby(group).transform(lambda s:s.rolling(3,min_periods=1).mean())
        df["delta_rolling_max_3"] = dt.groupby(group).transform(lambda s:s.rolling(3,min_periods=1).max())
    for col in SIGNALS:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            prev = v.groupby(group).shift(1)
            df[f"{col}_changed"] = (v != prev).astype(float)
            df.loc[prev.isna(), f"{col}_changed"] = 0.0
    for col in df.columns:
        if col != "_batch":
            df[col] = pd.to_numeric(df[col], errors="ignore")
    print(f"Loaded {batch}: {len(df)} rows; labels={df['anomaly'].value_counts().sort_index().to_dict()}")
    return df

def get_features(df, cols=None):
    blocked = EXCLUDED | {"_batch"}
    if cols is None:
        cols = [c for c in df.columns if c not in blocked and
                (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.replace([np.inf,-np.inf],np.nan), cols

def model(seed):
    return Pipeline([("imputer",SimpleImputer(strategy="median")),
        ("rf",RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",
             min_samples_leaf=2,max_features="sqrt",random_state=seed,n_jobs=-1))])

def choose_threshold(y, probs, max_fpr=0.10):
    rows=[]
    for t in np.arange(.05,.951,.025):
        pred=(probs>=t).astype(int)
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fpr=fp/(fp+tn) if fp+tn else 0
        rows.append({"threshold":round(float(t),3),"validation_fpr":fpr,
                     "validation_precision":precision_score(y,pred,zero_division=0),
                     "validation_recall":recall_score(y,pred,zero_division=0),
                     "validation_f1":f1_score(y,pred,zero_division=0)})
    scores=pd.DataFrame(rows)
    eligible=scores[scores.validation_fpr<=max_fpr]
    if len(eligible):
        best=eligible.sort_values(["validation_f1","validation_precision","threshold"],
                                  ascending=[False,False,False]).iloc[0]
    else:
        best=scores.sort_values(["validation_fpr","validation_f1"],ascending=[True,False]).iloc[0]
    return float(best.threshold),scores,best

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}. Edit DATASET_DIR at top of script.")
    data={b:load_batch(b) for b in BATCHES}
    train=pd.concat([df for b,df in data.items() if b not in {VALIDATION_BATCH,TEST_BATCH,BASELINE_BATCH}],ignore_index=True)
    val,test,base=data[VALIDATION_BATCH],data[TEST_BATCH],data[BASELINE_BATCH]
    Xtr,cols=get_features(train); Xv,_=get_features(val,cols); Xt,_=get_features(test,cols); Xb,_=get_features(base,cols)
    ytr=(train.anomaly!=0).astype(int); yv=(val.anomaly!=0).astype(int)
    yt=(test.anomaly!=0).astype(int); yb=(base.anomaly!=0).astype(int)
    detector=model(42); detector.fit(Xtr,ytr)
    classes=list(detector.named_steps["rf"].classes_)
    pv=detector.predict_proba(Xv)[:,classes.index(1)]
    threshold,scores,best=choose_threshold(yv,pv)
    pt=detector.predict_proba(Xt)[:,classes.index(1)]
    pb=detector.predict_proba(Xb)[:,classes.index(1)]
    detected=(pt>=threshold).astype(int); base_pred=(pb>=threshold).astype(int)
    print("\n===== STAGE 1: BINARY DETECTION =====")
    print(f"Selected threshold: {threshold:.3f}; validation FPR: {best.validation_fpr:.4f}; validation F1: {best.validation_f1:.4f}")
    print(f"Test accuracy={accuracy_score(yt,detected):.4f}; precision={precision_score(yt,detected,zero_division=0):.4f}; recall={recall_score(yt,detected,zero_division=0):.4f}; F1={f1_score(yt,detected,zero_division=0):.4f}")
    print("Test confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(yt,detected,labels=[0,1]))
    false_alarms=int(((base_pred==1)&(yb==0)).sum())
    print(f"Baseline false alarms: {false_alarms}/{len(base)} ({false_alarms/len(base):.2%})")

    anomaly_rows=train.anomaly!=0
    Xtype=Xtr.loc[anomaly_rows.to_numpy()]
    ytype=train.loc[anomaly_rows,"anomaly"].astype(int)
    type_model=model(43); type_model.fit(Xtype,ytype)
    type_pred=type_model.predict(Xt)
    actual=test.anomaly.astype(int).to_numpy()
    end_to_end=np.where(detected==1,type_pred,0).astype(int)
    labels=[0,1,2,3,4]; names=[NAMES[x] for x in labels]
    print("\n===== TWO-STAGE END-TO-END TEST RESULTS =====")
    print("Rows=actual categories; columns=predicted categories")
    print(pd.DataFrame(confusion_matrix(actual,end_to_end,labels=labels),
        index=[f"Actual: {NAMES[x]}" for x in labels],
        columns=[f"Pred: {NAMES[x]}" for x in labels]).to_string())
    print(classification_report(actual,end_to_end,labels=labels,target_names=names,zero_division=0,digits=4))
    mask=actual!=0
    print("\n===== FAULT-TYPE-ONLY DIAGNOSTIC =====")
    print("This diagnostic assumes an event is already known to be anomalous.")
    print(classification_report(actual[mask],type_pred[mask],labels=[1,2,3,4],
        target_names=[NAMES[x] for x in [1,2,3,4]],zero_division=0,digits=4))
    out=test.copy()
    out["stage1_anomaly_probability"]=pt
    out["stage1_predicted_anomaly"]=detected
    out["stage2_predicted_fault_type"]=type_pred
    out["end_to_end_prediction"]=end_to_end
    out["actual_anomaly_name"]=[NAMES[int(x)] for x in actual]
    out["end_to_end_prediction_name"]=[NAMES[int(x)] for x in end_to_end]
    p1=PROJECT_DIR/"daypsci_two_stage_test_predictions.csv"
    out.to_csv(p1,index=False,encoding="utf-8-sig")
    scores.to_csv(PROJECT_DIR/"daypsci_two_stage_threshold_scores.csv",index=False,encoding="utf-8-sig")
    report=classification_report(actual,end_to_end,labels=labels,target_names=names,zero_division=0,digits=4,output_dict=True)
    pd.DataFrame(report).transpose().to_csv(PROJECT_DIR/"daypsci_two_stage_test_report.csv",encoding="utf-8-sig")
    print("\nSaved outputs:")
    print(p1)
    print(PROJECT_DIR/"daypsci_two_stage_threshold_scores.csv")
    print(PROJECT_DIR/"daypsci_two_stage_test_report.csv")
    print("\nRare fault categories have few examples, so their metrics may be unstable. This is an experimental benchmark, not a safety-certified industrial system.")

if __name__=="__main__":
    main()
