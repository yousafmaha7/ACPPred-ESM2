# app.py
import streamlit as st
import torch
import pandas as pd
import joblib
from io import StringIO
from Bio import SeqIO
from esm.pretrained import esm2_t33_650M_UR50D

# Load trained AdaBoost model (trained on 1280-dimensional ESM-2 embeddings)
model = joblib.load("best_adaboost_esm2_model.pkl")

# Load pretrained ESM-2 model: esm2_t33_650M_UR50D gives 1280-d features
#esm_model, alphabet = esm2_t33_650M_UR50D()
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# Function to extract ESM-2 features
def extract_esm_features(sequences):
    batch_labels = [f"seq{i+1}" for i in range(len(sequences))]
    batch_data = list(zip(batch_labels, sequences))
    _, _, batch_tokens = batch_converter(batch_data)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    features = []
    for i, (_, seq) in enumerate(batch_data):
        # Mean pooling excluding [CLS] and [EOS]
        rep = token_representations[i, 1:len(seq)+1].mean(0).numpy()
        features.append(rep)
    return features

# Streamlit UI
st.title("ACP-ESM2: Tool for Anticancer Peptide Prediction")

input_method = st.radio("Choose input method:", ["Paste Sequence", "Upload FASTA File"])
sequences = []

# Input handling
if input_method == "Paste Sequence":
    seq_text = st.text_area("Enter peptide sequence(s), one per line:")
    if seq_text:
        sequences = [line.strip() for line in seq_text.strip().split("\n") if line.strip()]

elif input_method == "Upload FASTA File":
    uploaded_file = st.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        fasta_io = StringIO(content)
        sequences = [str(record.seq) for record in SeqIO.parse(fasta_io, "fasta")]

# Prediction and results
if sequences:
    st.write(f"Total sequences: {len(sequences)}")

    if st.button("Predict"):
        with st.spinner("Extracting features and predicting..."):
            try:
                features = extract_esm_features(sequences)
                preds = model.predict(features)
                df = pd.DataFrame({
                    "Sequence": sequences,
                    "Prediction": ["Positive" if p == 1 else "Negative" for p in preds]
                })
                st.success("Prediction complete!")
                st.dataframe(df)

                # Download button
                csv = df.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"An error occurred during prediction:\n{str(e)}")
