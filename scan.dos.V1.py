import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
from datetime import datetime
from plyfile import PlyData
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# ==============================
# CONFIG & STYLE
# ==============================
st.set_page_config(page_title="SpineScan Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { border-left: 5px solid #007bff; background: white; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# FONCTIONS TECHNIQUES
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    vertex = plydata['vertex']
    return np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

def compute_cobb(x, y):
    dy, dx = np.gradient(y), np.gradient(x)
    slopes = dx / (dy + 1e-6)
    angle = np.degrees(abs(np.arctan(slopes.max()) - np.arctan(slopes.min())))
    return round(angle, 1)

# ==============================
# SIDEBAR (Paramètres de lissage)
# ==============================
with st.sidebar:
    st.header("📋 Patient & Setup")
    nom = st.text_input("Nom", "Anonyme")
    prenom = st.text_input("Prénom", "")
    st.divider()
    
    st.subheader("🛠 Paramètres IA")
    do_smooth = st.toggle("Activer le lissage", True)
    smooth_win = st.slider("Fenêtre de lissage (Savgol)", 5, 51, 21, step=2)
    k_std = st.slider("Filtre bruit (Std Dev)", 0.5, 3.0, 1.5)
    
    ply_file = st.file_uploader("Fichier PLY", type=["ply"])

# ==============================
# ANALYSE ET RENDU
# ==============================
if ply_file and st.button("🚀 Lancer le bilan complet"):
    # 1. Chargement et normalisation
    pts = load_ply_numpy(ply_file) * 0.1 # cm
    
    # 2. Extraction du rachis par tranches
    slices = np.linspace(pts[:,1].min(), pts[:,1].max(), 60)
    spine_points = []
    
    for i in range(len(slices)-1):
        mask = (pts[:,1] >= slices[i]) & (pts[:,1] < slices[i+1])
        sl = pts[mask]
        if len(sl) > 10:
            mx, sx = sl[:,0].mean(), sl[:,0].std()
            clean_sl = sl[(sl[:,0] > mx - k_std*sx) & (sl[:,0] < mx + k_std*sx)]
            if len(clean_sl) > 0:
                spine_points.append([clean_sl[:,0].mean(), clean_sl[:,1].mean(), clean_sl[:,2].mean()])
    
    spine = np.array(spine_points)
    
    # 3. RÉAPPLIQUER LE LISSAGE (ICI)
    if do_smooth and len(spine) > smooth_win:
        # Savitzky-Golay : (données, fenêtre, polynôme)
        spine[:,0] = savgol_filter(spine[:,0], list(filter(lambda x: x < len(spine), [smooth_win]))[0], 3)
        spine[:,2] = savgol_filter(spine[:,2], list(filter(lambda x: x < len(spine), [smooth_win]))[0], 3)

    # 4. Métriques
    cobb = compute_cobb(spine[:,0], spine[:,1])
    f_dorsale = round(abs(spine[:,2].max() - spine[:,2].mean()), 2)
    f_lombaire = round(abs(spine[:,2].min() - spine[:,2].mean()), 2)

    # 5. Affichage Dashboard
    st.subheader(f"Analyse de {prenom} {nom}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Angle de Cobb", f"{cobb}°")
    c2.metric("Flèche Dorsale", f"{f_dorsale} cm")
    c3.metric("Flèche Lombaire", f"{f_lombaire} cm")

    # 6. Graphiques
    tmp = tempfile.gettempdir()
    p_front, p_side = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Vue Frontale
    ax1.scatter(pts[:,0], pts[:,1], s=0.5, alpha=0.1, color='gray')
    ax1.plot(spine[:,0], spine[:,1], 'red', linewidth=2, label="Axe rachidien")
    ax1.set_title("Vue Frontale (Scoliose)")
    ax1.legend()
    
    # Vue Sagittale
    ax2.scatter(pts[:,2], pts[:,1], s=0.5, alpha=0.1, color='gray')
    ax2.plot(spine[:,2], spine[:,1], 'blue', linewidth=2, label="Profil")
    ax2.set_title("Vue Sagittale (Cyphose/Lordose)")
    
    st.pyplot(fig)
    fig.savefig(p_front) # Utilisation simplifiée pour le PDF

    # 7. PDF Pro (Version condensée)
    # [Logique de génération PDF identique à la précédente mais avec les données lissées]
    st.success("Analyse terminée avec lissage des courbes.")
