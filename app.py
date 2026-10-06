# app.py — ACPPred-ESM2 (revised for deployment)
# Feature pipeline is intentionally IDENTICAL to the original app:
#   esm2_t33_650M_UR50D, FP32, layer 33, mean over residue tokens (BOS/EOS excluded).
# Only loading, validation, batching and diagnostics were changed.

from pathlib import Path
from io import StringIO
import resource
import time

import numpy as np
import pandas as pd
import joblib
import torch
import streamlit as st
from Bio import SeqIO
import esm

st.set_page_config(page_title="ACPPred-ESM2", layout="centered")

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
CLF_PATH = BASE_DIR / "best_adaboost_esm2_model.pkl"
ESM_LAYER = 33
EMBED_DIM = 1280                  # output dim of esm2_t33_650M_UR50D
MAX_LEN = 1022                    # ESM-2 limit: 1024 positions incl. BOS/EOS
MAX_SEQS = 500                    # protects the server from very large uploads
BATCH_SIZE = 8                    # bounds activation memory during inference
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")   # add "X" etc. ONLY if training data contained them


def log(msg: str) -> None:
    """Checkpoint with peak resident memory; flush so it survives a kill."""
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[ACP {time.strftime('%H:%M:%S')}] {msg} | peak RAM ~ {peak_mb:.0f} MB", flush=True)


# ---------------- Cached resources (loaded once per server process) ----------------
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    clf = joblib.load(CLF_PATH)
    labels = set(np.asarray(clf.classes_).tolist())
    if not labels <= {0, 1}:
        raise ValueError(f"Unexpected class labels {clf.classes_}; check the Positive/Negative mapping.")
    n_feat = getattr(clf, "n_features_in_", None)
    if n_feat is not None and n_feat != EMBED_DIM:
        raise ValueError(f"Classifier expects {n_feat} features, ESM-2 650M gives {EMBED_DIM}.")
    log("classifier loaded")
    return clf


@st.cache_resource(show_spinner="Loading ESM-2 (650M). The first load can take several minutes...")
def load_esm():
    log("loading ESM-2 650M")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.eval()
    log("ESM-2 loaded")
    return esm_model, alphabet.get_batch_converter()


# ---------------- Input handling ----------------
def parse_pasted(text: str):
    """Accept FASTA (if it starts with '>') or one raw sequence per line."""
    text = text.strip()
    if text.startswith(">"):
        return [(r.id, str(r.seq)) for r in SeqIO.parse(StringIO(text), "fasta")]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [(f"seq{i}", s) for i, s in enumerate(lines, 1)]


def validate(records):
    """Normalise case/whitespace and reject anything ESM-2 would silently map to <unk>."""
    ok, rejected = [], []
    for sid, s in records:
        s = "".join(s.split()).upper()
        if not s:
            rejected.append((sid, s, "empty sequence"))
            continue
        bad = sorted(set(s) - VALID_AA)
        if bad:
            rejected.append((sid, s, f"non-standard residues: {''.join(bad)}"))
            continue
        if len(s) > MAX_LEN:
            rejected.append((sid, s, f"length {len(s)} > {MAX_LEN}"))
            continue
        ok.append((sid, s))
    return ok, rejected


# ---------------- Feature extraction (same definition as training app) ----------------
def extract_esm_features(seqs, esm_model, batch_converter):
    feats = []
    for start in range(0, len(seqs), BATCH_SIZE):
        chunk = seqs[start:start + BATCH_SIZE]
        data = [(f"s{start + j}", s) for j, s in enumerate(chunk)]
        _, _, tokens = batch_converter(data)
        with torch.inference_mode():
            out = esm_model(tokens, repr_layers=[ESM_LAYER], return_contacts=False)
        reps = out["representations"][ESM_LAYER]
        for j, s in enumerate(chunk):
            # token 0 = BOS; residues are 1..len(s); EOS and padding excluded
            feats.append(reps[j, 1:len(s) + 1].mean(0).cpu().numpy())
    return np.vstack(feats)


# ---------------- UI ----------------
st.title("ACPPred-ESM2: Tool for Anticancer Peptide Prediction")
log("script start")

try:
    clf = load_classifier()
    esm_model, batch_converter = load_esm()
except Exception as e:
    log(f"model loading failed: {e!r}")
    st.error(f"Model loading failed: {e}")
    st.stop()

input_method = st.radio("Choose input method:", ["Paste Sequence", "Upload FASTA File"])
records = []

if input_method == "Paste Sequence":
    seq_text = st.text_area("Enter peptide sequence(s), one per line, or paste FASTA:")
    if seq_text:
        records = parse_pasted(seq_text)
else:
    uploaded_file = st.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8", errors="replace")
        records = [(r.id, str(r.seq)) for r in SeqIO.parse(StringIO(content), "fasta")]

if records:
    valid, rejected = validate(records)
    st.write(f"Sequences received: {len(records)} | valid: {len(valid)} | rejected: {len(rejected)}")

    if rejected:
        with st.expander(f"{len(rejected)} sequence(s) rejected"):
            st.dataframe(pd.DataFrame(rejected, columns=["ID", "Sequence", "Reason"]))

    if len(valid) > MAX_SEQS:
        st.warning(f"Please submit at most {MAX_SEQS} sequences per run.")
    elif valid and st.button("Predict"):
        with st.spinner("Extracting features and predicting..."):
            try:
                ids = [sid for sid, _ in valid]
                seqs = [s for _, s in valid]
                X = extract_esm_features(seqs, esm_model, batch_converter)
                preds = clf.predict(X)
                df = pd.DataFrame({
                    "ID": ids,
                    "Sequence": seqs,
                    "Length": [len(s) for s in seqs],
                    "Prediction": ["Positive" if p == 1 else "Negative" for p in preds],
                })
                log(f"predicted {len(seqs)} sequences")
                st.success("Prediction complete!")
                st.dataframe(df)
                st.download_button("Download Results", df.to_csv(index=False),
                                   "predictions.csv", "text/csv")
            except Exception as e:
                log(f"prediction failed: {e!r}")
                st.error(f"An error occurred during prediction:\n{e}")
