import zipfile
import os

# Création du dossier du projet
folder_name = "mini_projet_sar"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Contenu de app.py
app_code = """import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from PIL import Image
import torch
import torch.nn as nn

st.set_page_config(page_title="Mini Projet SAR", layout="wide")
st.title("Mini-Projet : Interférométrie SAR - Filtrage Classique vs IA")

uploaded_file = st.file_uploader("Importer une image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = np.array(image) / 255.0
    N = image.shape[0]

    SNR_dB = st.slider("SNR (dB)", 0, 20, 5)
    SNR = 10**(SNR_dB/10)

    phase = 4*np.pi*image
    s1 = image * np.exp(1j*phase)

    deform = 0.2*np.sin(2*np.pi*np.linspace(0,1,N))
    deform = np.tile(deform,(N,1))
    s2_clean = image * np.exp(1j*(phase + deform))

    noise1 = (np.random.randn(N,N)+1j*np.random.randn(N,N))/np.sqrt(2*SNR)
    noise2 = (np.random.randn(N,N)+1j*np.random.randn(N,N))/np.sqrt(2*SNR)
    s1_noisy = s1 + noise1
    s2_noisy = s2_clean + noise2

    def coherence(s1, s2):
        num = uniform_filter(s1*np.conj(s2), 5)
        den = np.sqrt(uniform_filter(np.abs(s1)**2, 5) *
                      uniform_filter(np.abs(s2)**2, 5))
        return np.abs(num/(den+1e-8))

    gamma_raw = coherence(s1_noisy, s2_noisy)

    s1_filt = uniform_filter(s1_noisy, 3)
    s2_filt = uniform_filter(s2_noisy, 3)
    gamma_filt = coherence(s1_filt, s2_filt)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2,16,3,padding=1),
                nn.ReLU(),
                nn.Conv2d(16,2,3,padding=1)
            )
        def forward(self,x):
            return self.net(x)

    model = SimpleCNN()

    input_tensor = np.stack([np.real(s1_noisy), np.imag(s1_noisy)], axis=0)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).numpy()[0]

    s1_dnn = output[0] + 1j*output[1]
    gamma_dnn = coherence(s1_dnn, s2_noisy)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image originale", use_column_width=True)
    with col2:
        st.write("Cohérence moyenne brute :", np.mean(gamma_raw))
        st.write("Cohérence moyenne filtrage classique :", np.mean(gamma_filt))
        st.write("Cohérence moyenne filtrage IA :", np.mean(gamma_dnn))

    fig, axs = plt.subplots(1,3, figsize=(18,5))
    axs[0].imshow(gamma_raw, cmap='jet'); axs[0].set_title("Interférométrie brute")
    axs[1].imshow(gamma_filt, cmap='jet'); axs[1].set_title("Filtrage classique")
    axs[2].imshow(gamma_dnn, cmap='jet'); axs[2].set_title("Filtrage IA")
    st.pyplot(fig)
"""

# Contenu requirements.txt
requirements = """streamlit
numpy
scipy
matplotlib
torch
pillow
"""

# Contenu README.md
readme = """# Mini-Projet SAR/InSAR

## Installation

1. Cloner le dépôt
