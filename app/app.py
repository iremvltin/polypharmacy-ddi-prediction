import streamlit as st
import pandas as pd
import numpy as np
import itertools
import joblib
import os
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import shap
import matplotlib.pyplot as plt

# --- PUBCHEM API ---
@st.cache_data(ttl=3600, show_spinner="Looking up drug information...")
def get_smiles_from_pubchem(drug_name):
    if not drug_name or len(drug_name.strip()) == 0:
        return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name.strip()}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            props = response.json()['PropertyTable']['Properties'][0]
            return props.get('CanonicalSMILES') or props.get('ConnectivitySMILES')
        return None
    except Exception:
        return None

st.set_page_config(page_title="Polypharmacy DDI Prediction System", layout="wide")

st.markdown("""
<style>
    /* Make the text inputs take full width of the main page layout */
    .stTextInput input {
        width: 100% !important;
    }
    /* Balanced Custom Cards that match standard web framework inputs */
    .risk-card {
        border: 1px solid #dcdfe6;
        background-color: #f5f7fa; 
        padding: 20px; 
        border-radius: 8px; 
        margin-bottom: 15px;
        color: #303133;
    }
    .risk-card h4 {
        margin-top: 0px;
        color: #2f3542;
    }
    /* Style for the prominent methodological warning */
    .warning-card {
        border-left: 4px solid #e6a23c;
        background-color: #fdf6ec;
        padding: 15px;
        border-radius: 4px;
        margin-top: 15px;
        margin-bottom: 25px;
        color: #606266;
    }
</style>
""", unsafe_allow_html=True)

st.title("Polypharmacy Drug-Drug Interaction (DDI) Prediction System")
st.write("This platform predicts potential binary interaction risks when multiple medications are administered simultaneously.")

# --- CRITICAL METHODOLOGICAL WARNING ---
st.markdown("""
<div class="warning-card">
    <strong>CRITICAL NOTICE & LIMITATIONS:</strong><br>
    1. This platform is designed strictly as a supplementary decision-support tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.<br>
    2. <strong>Scope Limitation:</strong> This system evaluates interactions on a <em>pairwise (binary)</em> basis. Higher-order, complex combinations (e.g., ternary or higher order <strong>emergent interactions</strong> involving three or more drugs simultaneously) cannot be captured by this pairwise machine learning approach.
</div>
""", unsafe_allow_html=True)

# --- 2. ASSETS LOADING ---
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'logistic_regression_model.pkl')
    
    model = joblib.load(MODEL_PATH)
    
    num_features = getattr(model, "n_features_in_", 4096)
    background_data = np.zeros((10, num_features)) 
    explainer = shap.LinearExplainer(model, background_data)
    
    return model, explainer

try:
    model1, shap_explainer = load_assets()
    st.success("AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"Failed to load model assets! Ensure 'logistic_regression_model.pkl' is in the root directory. Error: {e}")
    st.stop()

# --- 3. FINGERPRINT GENERATION FUNCTION ---
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- 4. MAIN PAGE CONTROLS & INPUTS ---
st.subheader("Medication List Management")

num_drugs = st.number_input("Select number of medications to analyze (2-10):", min_value=2, max_value=10, value=3)

drugs = []
for i in range(int(num_drugs)):
    st.markdown(f"**Medication #{i+1}**")

    drug_name = st.text_input(f"Enter Drug Name #{i+1}", key=f"drug_name_{i}")

    smiles_val = None
    if drug_name:
        smiles_val = get_smiles_from_pubchem(drug_name)
        if smiles_val:
            st.caption(f"✓ '{drug_name}' identified successfully.")
        else:
            st.warning(f"Could not find a matching chemical structure for '{drug_name}' on PubChem. Please check the spelling (English drug names work best).")

    if drug_name and smiles_val:
        drugs.append({"name": drug_name, "smiles": smiles_val})

# --- 5. PREDICTION AND COMBINATION FLOW ---
st.markdown("---")
if len(drugs) < 2:
    st.info("Please provide valid names for at least 2 medications on the list above.")
else:
    st.subheader(f"Combination Analysis for {len(drugs)} Medications")
    
    if st.button("Analyze Interaction Risks", type="primary"):
        combinations = list(itertools.combinations(drugs, 2))
        risk_found = False
        
        for d1, d2 in combinations:
            fp1 = smiles_to_fp(d1["smiles"])
            fp2 = smiles_to_fp(d2["smiles"])
            if fp1 is None or fp2 is None: 
                st.warning(f"Invalid SMILES string detected for {d1['name']} or {d2['name']}. Skipping this pair.")
                continue
            
            features = np.concatenate([fp1, fp2]).reshape(1, -1)
            prediction = model1.predict(features)[0]
            probability = model1.predict_proba(features)[0][1]
            
            if prediction == 1 or probability >= 0.5:
                risk_found = True
                risk_level = "HIGH RISK" if probability > 0.85 else "MODERATE RISK"
                
                st.markdown(f"""
                <div class="risk-card">
                    <h4>Interaction Risk Detected: {d1['name']} + {d2['name']}</h4>
                    <p><b>Risk Status:</b> {risk_level} | <b>Interaction Probability:</b> {probability*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"Explainable AI (SHAP) Analysis for {d1['name']} + {d2['name']}"):
                    shap_values = shap_explainer.shap_values(features)
                    fig, ax = plt.subplots(figsize=(10, 2))
                    shap.plots.bar(shap.Explanation(values=shap_values[0], data=features[0]), max_display=10, show=False)
                    plt.title("Top 10 Chemical Bit Features Influencing the Prediction", fontsize=10)
                    st.pyplot(fig)
                    plt.close()
                    
        if not risk_found:
            st.success("Analysis complete. No adverse interaction risks were detected among the evaluated combinations.")

