# ACP-ESM2: Machine Learning and Transfer Learning based Anticancer Peptide Prediction Web Tool
ACP-ESM2 is novel, state of the art, machine learning and transfer learning based web application that employs a pretrained ESM-2 model and AdaBoost classifier to perform anticancer peptide (ACP) classification.
## Features

- Accepts peptide sequences in FASTA format
- Extracts ESM-2 embeddings
- Applies a trained AdaBoost model to classify sequences
- Interactive Streamlit-based UI

## 🔧 Installation (Local)

To run this app locally, follow these steps:

### 1. Clone the Repository
```
git clone https://github.com/yousafmaha7/ACP-ESM2.git
cd acp-esm2
```
### 2. Create and Activate a Virtual Environment (Recommended)
If you are using windows, then open command prompt or windows powershell in your desired directory.
If you are using Linux, then open terminal in your desired directory
```
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 4. Run the App
```
streamlit run app.py
```
