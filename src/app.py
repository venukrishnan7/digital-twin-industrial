import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Digital Twin Validation Dashboard", page_icon="🏭", layout="wide")

st.title(" Digital Twin-Driven Validation Dashboard")
st.caption("Sequential control logic • Random Forest fault detection • Hybrid rule override")
st.warning(
    "Research prototype using simulated/perturbed sequences. It is not connected to a physical PLC "
    "and must not be used as a safety system."
)

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "results"

def read_csv_upload_or_path(uploaded, path):
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            return None
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read {path.name}: {e}")
    return None

def read_text_upload_or_path(uploaded, path):
    if uploaded is not None:
        return uploaded.getvalue().decode("utf-8", errors="replace")
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def to_num(series):
    return pd.to_numeric(series, errors="coerce")

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

with st.sidebar:
    st.header("Data sources")
    st.write("Upload your existing experiment CSVs, or place them in a `results` folder beside `app.py`.")
    st.caption("No new model is trained by this dashboard. It visualizes saved experiment outputs.")
    dataset_upload = st.file_uploader("Base Phase 6 dataset (CSV)", type=["csv"], key="base")
    pred6b_upload = st.file_uploader("Phase 6B predictions (CSV)", type=["csv"], key="p6b")
    pred6c_upload = st.file_uploader("Phase 6C hybrid predictions (CSV)", type=["csv"], key="p6c")
    stress_upload = st.file_uploader("Phase 6E stress predictions (CSV)", type=["csv"], key="stress")
    st.divider()
    st.caption("Expected local filenames:")
    st.code(
        "results/factory_phase6_dataset.csv\n"
        "results/factory_phase6b_fault_predictions.csv\n"
        "results/factory_phase6c_hybrid_predictions.csv\n"
        "results/factory_phase6e_stress_predictions.csv",
        language="text"
    )


base_df = read_csv_upload_or_path(
    dataset_upload,
    DEFAULT_DIR / "processed" / "factory_phase6_dataset.csv"
)

p6b_df = read_csv_upload_or_path(
    pred6b_upload,
    DEFAULT_DIR / "predictions" / "factory_phase6b_fault_predictions.csv"
)

p6c_df = read_csv_upload_or_path(
    pred6c_upload,
    DEFAULT_DIR / "predictions" / "factory_phase6c_hybrid_predictions.csv"
)

stress_df = read_csv_upload_or_path(
    stress_upload,
    DEFAULT_DIR / "predictions" / "factory_phase6e_stress_predictions.csv"
)


tabs = st.tabs(["Overview", "Live-style Twin", "Model Results", "Stress Test", "Data Explorer"])

with tabs[0]:
    st.subheader("Experiment overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project phase", "5C–6E")
    c2.metric("Classifier", "Random Forest")
    c3.metric("Hybrid logic", "RF OR rules")
    c4.metric("Data type", "Simulated")
    if base_df is not None:
        st.success(f"Base dataset loaded: {len(base_df):,} event rows and {base_df['run_id'].nunique():,} runs." if "run_id" in base_df.columns else f"Base dataset loaded: {len(base_df):,} rows.")
        st.write("Columns detected:")
        st.code(", ".join(map(str, base_df.columns)))
    else:
        st.info("Upload your base dataset in the sidebar or copy it to the `results` folder. The dashboard can still show the reported summary metrics below.")
    st.markdown("### Reported results from your completed experiments")
    rows = [
        ["Phase 6B", "RF baseline", "95.94%", "96.09%", "96.35%", "0.9103"],
        ["Phase 6C", "Hybrid holdout", "96.72%", "97.92%", "100.00%", "0.9288"],
        ["Phase 6D", "RF repeated-split mean", "95.32%", "96.06%", "97.64%", "0.9131"],
        ["Phase 6D", "Hybrid repeated-split mean", "95.87%", "97.24%", "100.00%", "0.9252"],
        ["Phase 6E", "RF perturbed stress test", "88.35%", "88.68%", "77.35%", "0.8723"],
        ["Phase 6E", "Hybrid perturbed stress test", "90.93%", "91.19%", "82.37%", "0.9033"],
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Phase", "Method", "Accuracy", "Balanced accuracy", "Fault recall", "F1-score"]), use_container_width=True, hide_index=True)
    st.caption("Reported values are the experiment summaries previously provided. Upload prediction files to inspect row-level outputs. Verify against your archived metrics files before final submission.")

with tabs[1]:
    st.subheader("Sequential process — live-style demonstration")
    st.write("This is a dashboard-side demonstration of the state sequence, not a live PLC connection.")
    scenario = st.selectbox("Choose a simulated scenario", [
        "Normal operation", "Slow drilling", "Part-sensor failure", "Clamp interlock"
    ])
    states = ["IDLE", "CONVEYING", "CLAMPING", "DRILLING", "UNCLAMPING", "RELEASING", "COMPLETE"]
    if scenario == "Normal operation":
        state_seq = states
        status, detail = "NORMAL", "Expected sequence reaches COMPLETE."
        color = "green"
    elif scenario == "Slow drilling":
        state_seq = ["IDLE", "CONVEYING", "CLAMPING", "DRILLING (SLOW)", "DRILLING (TIMEOUT)", "FAULT"]
        status, detail, color = "FAULT ALERT", "Drilling exceeded the expected time threshold in this demonstration.", "red"
    elif scenario == "Part-sensor failure":
        state_seq = ["IDLE", "CONVEYING", "WAITING FOR PART", "SENSOR TIMEOUT", "FAULT"]
        status, detail, color = "FAULT ALERT", "Part-detection confirmation was not received in time.", "red"
    else:
        state_seq = ["IDLE", "CONVEYING", "CLAMPING", "INTERLOCK BLOCKED", "FAULT"]
        status, detail, color = "SAFETY INTERLOCK", "The clamp interlock prevents normal sequence progression.", "red"
    if color == "green":
        st.success(f"Status: {status} — {detail}")
    else:
        st.error(f"Status: {status} — {detail}")
    cols = st.columns(min(len(state_seq), 4))
    for i, state in enumerate(state_seq):
        with cols[i % len(cols)]:
            st.markdown(f"**Step {i+1}**")
            st.info(state) if state not in ["FAULT", "SENSOR TIMEOUT", "INTERLOCK BLOCKED", "DRILLING (TIMEOUT)"] else st.error(state)
    st.caption("The state sequence above is illustrative UI logic. For an evidence-based run, execute your Phase 5C simulator and show its actual console/CSV output alongside this dashboard.")

with tabs[2]:
    st.subheader("Saved model predictions")
    pred_df = p6b_df if p6b_df is not None else p6c_df
    if pred_df is None:
        st.info("Upload Phase 6B or Phase 6C predictions to inspect the actual row-level predictions.")
    else:
        st.write(f"Loaded {len(pred_df):,} prediction rows.")
        st.write("Detected columns:", ", ".join(map(str, pred_df.columns)))
        st.dataframe(pred_df.head(100), use_container_width=True)
        st.download_button("Download displayed prediction data", pred_df.to_csv(index=False).encode("utf-8"), "predictions_preview.csv", "text/csv")
        pred_col = find_col(pred_df, ["fault_prediction", "y_pred", "prediction", "rf_prediction", "predicted_fault", "rf_pred"])
        actual_col = find_col(pred_df, ["fault_active_sim", "y_true", "actual", "actual_fault", "target", "fault_label"])
        if pred_col and actual_col:
            ypred = to_num(pred_df[pred_col])
            ytrue = to_num(pred_df[actual_col])
            valid = ypred.notna() & ytrue.notna()
            if valid.any():
                st.markdown("#### Metrics recalculated from detected columns")
                from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                yt, yp = ytrue[valid].astype(int), ypred[valid].astype(int)
                a,b,c,d,e = st.columns(5)
                metric_card("Accuracy", f"{accuracy_score(yt,yp):.2%}")
                metric_card("Balanced accuracy", f"{balanced_accuracy_score(yt,yp):.2%}")
                metric_card("Fault precision", f"{precision_score(yt,yp,zero_division=0):.2%}")
                metric_card("Fault recall", f"{recall_score(yt,yp,zero_division=0):.2%}")
                metric_card("F1-score", f"{f1_score(yt,yp,zero_division=0):.4f}")
                cm = confusion_matrix(yt, yp, labels=[0,1])
                st.write("Confusion matrix — rows actual [0,1], columns predicted [0,1]")
                st.dataframe(pd.DataFrame(cm, index=["Actual inactive","Actual fault"], columns=["Predicted inactive","Predicted fault"]), use_container_width=True)
        else:
            st.caption("Automatic metric recalculation was skipped because standard actual/prediction column names were not detected. Inspect the columns above and adapt the aliases in app.py if needed.")

with tabs[3]:
    st.subheader("Perturbed-sequence stress test")
    if stress_df is None:
        st.info("Upload `factory_phase6e_stress_predictions.csv` to view the actual stress-test rows.")
        stress_rows = [
            ["RF", "88.35%", "88.68%", "100.00%", "77.35%", "0.8723", "212", "0"],
            ["Hybrid", "90.93%", "91.19%", "100.00%", "82.37%", "0.9033", "165", "0"],
        ]
        st.dataframe(pd.DataFrame(stress_rows, columns=["Method","Accuracy","Balanced accuracy","Precision","Recall","F1-score","Missed faults","False alarms"]), use_container_width=True, hide_index=True)
    else:
        st.success(f"Stress prediction file loaded: {len(stress_df):,} rows.")
        st.dataframe(stress_df.head(100), use_container_width=True)
        st.download_button("Download stress predictions", stress_df.to_csv(index=False).encode("utf-8"), "stress_predictions_download.csv", "text/csv")
        case_col = find_col(stress_df, ["stress_case", "scenario", "case", "stress_scenario"])
        actual_col = find_col(stress_df, ["fault_active_sim", "y_true", "actual", "actual_fault", "target", "fault_label"])
        rf_col = find_col(stress_df, ["rf_prediction", "rf_pred", "rf_prediction_label", "prediction"])
        hy_col = find_col(stress_df, ["hybrid_prediction", "hybrid_pred", "hybrid_prediction_label"])
        if case_col:
            st.markdown("#### Rows by stress case")
            st.bar_chart(stress_df[case_col].value_counts())
            if actual_col and rf_col and hy_col:
                st.markdown("#### Recall by stress case")
                from sklearn.metrics import recall_score
                out=[]
                for case, grp in stress_df.groupby(case_col):
                    y=to_num(grp[actual_col])
                    for method, col in [("RF",rf_col),("Hybrid",hy_col)]:
                        p=to_num(grp[col])
                        m=y.notna() & p.notna()
                        if m.any() and y[m].nunique()>1:
                            out.append({"Stress case":case,"Method":method,"Fault recall":recall_score(y[m].astype(int),p[m].astype(int),zero_division=0)})
                if out:
                    chart_df=pd.DataFrame(out).pivot(index="Stress case",columns="Method",values="Fault recall")
                    st.bar_chart(chart_df)
        else:
            st.caption("Stress-case breakdown needs a column such as stress_case or scenario. Check your CSV column names.")

with tabs[4]:
    st.subheader("Data explorer")
    options = {name: df for name, df in [
        ("Base dataset", base_df), ("Phase 6B predictions", p6b_df),
        ("Phase 6C predictions", p6c_df), ("Phase 6E stress predictions", stress_df)
    ] if df is not None}
    if not options:
        st.info("No CSV data loaded yet. Use the sidebar upload controls.")
    else:
        chosen = st.selectbox("Dataset", list(options.keys()))
        df = options[chosen]
        q = st.text_input("Filter rows (text search across displayed values)", "")
        view = df
        if q.strip():
            mask = df.astype(str).apply(lambda col: col.str.contains(q, case=False, na=False)).any(axis=1)
            view = df[mask]
        st.caption(f"{len(view):,} rows shown from {len(df):,} total rows.")
        st.dataframe(view, use_container_width=True, height=420)
        st.download_button("Download selected dataset", view.to_csv(index=False).encode("utf-8"), "selected_dataset.csv", "text/csv")

st.divider()
st.caption("For academic demonstration only. This dashboard visualizes saved simulated experiments; it does not control equipment or guarantee safety.")
