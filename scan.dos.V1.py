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
# CONFIG & DESIGN
# ==============================
st.set_page_config(page_title="SpineScan Pro 3D", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fc; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    div.stMetric { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# FONCTIONS TECHNIQUES
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata['vertex']
    return np.vstack([v['x'], v['y'], v['z']]).T

def compute_cobb_angle(x, y):
    dy, dx = np.gradient(y), np.gradient(x)
    slopes = dx / (dy + 1e-6)
    return np.degrees(abs(np.arctan(slopes.max()) - np.arctan(slopes.min())))

def compute_sagittal_arrows(spine_cm):
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]
    # Ligne de référence entre C7 et S1 (haut et bas du scan)
    z_ref_line = np.linspace(z[0], z[-1], len(z))
    delta = z - z_ref_line
    f_dorsale = abs(np.max(delta))
    f_lombaire = abs(np.min(delta))
    return f_dorsale, f_lombaire, z_ref_line

# ==============================
# GÉNÉRATION PDF PROFESSIONNEL
# ==============================
def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spine_pro.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    header_s = ParagraphStyle('Header', fontSize=18, textColor=colors.HexColor("#2c3e50"), spaceAfter=20, alignment=1)
    body_s = ParagraphStyle('Body', fontSize=10, spaceBefore=6)

    story = []
    
    # En-tête
    story.append(Paragraph("<b>BILAN CLINIQUE DU RACHIS 3D</b>", header_s))
    story.append(Paragraph(f"Date de l'examen : {datetime.now().strftime('%d/%m/%Y')}", body_s))
    story.append(Paragraph(f"Patient : {patient_info['prenom']} {patient_info['nom']}", body_s))
    story.append(Spacer(1, 1*cm))

    # Tableau des mesures
    data = [
        ["Indicateur", "Mesure", "Référence"],
        ["Angle de Cobb", f"{results['cobb']:.1f}°", "< 10° (Normal)"],
        ["Flèche Dorsale", f"{results['fd']:.2f} cm", "3.0 - 5.0 cm"],
        ["Flèche Lombaire", f"{results['fl']:.2f} cm", "2.0 - 4.0 cm"],
        ["Déviation Frontale Max", f"{results['dev_f']:.2f} cm", "-"]
    ]
    t = Table(data, colWidths=[6*cm, 4*cm, 5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 1*cm))

    # Images
    img_t = Table([[PDFImage(img_f, width=7*cm, height=9*cm), PDFImage(img_s, width=7*cm, height=9*cm)]])
    story.append(img_t)
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("<i>Note : Ce document est une synthèse automatisée basée sur une analyse de surface 3D.</i>", styles['Italic']))
    
    doc.build(story)
    return path

# ==============================
# INTERFACE SIDEBAR
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "Anonyme")
    prenom = st.text_input("Prénom", "")
    st.divider()
    st.subheader("⚙️ Paramètres d'Analyse")
    do_smooth = st.toggle("Activer le lissage", True)
    smooth_val = st.slider("Intensité du lissage", 5, 51, 21, step=2)
    k_std = st.slider("Tolérance bruit", 0.5, 3.0, 1.5)
    ply_file = st.file_uploader("Charger Scan PLY", type=["ply"])

# ==============================
# MAIN APP
# ==============================
st.title("🦴 SpineScan Pro")

if ply_file:
    # --- Traitement des données ---
    pts = load_ply_numpy(ply_file) * 0.1 # Passage en cm
    
    # Filtrage et centrage
    mask = (pts[:,1] > np.percentile(pts[:,1], 5)) & (pts[:,1] < np.percentile(pts[:,1], 95))
    pts = pts[mask]
    pts[:,0] -= pts[:,0].mean()
    
    # Extraction du rachis
    slices = np.linspace(pts[:,1].min(), pts[:,1].max(), 60)
    spine = []
    for i in range(len(slices)-1):
        sl = pts[(pts[:,1]>=slices[i]) & (pts[:,1]<slices[i+1])]
        if len(sl) > 5:
            mx, sx = sl[:,0].mean(), sl[:,0].std()
            sl = sl[(sl[:,0] > mx - k_std*sx) & (sl[:,0] < mx + k_std*sx)]
            if len(sl) > 0:
                spine.append([sl[:,0].mean(), sl[:,1].mean(), sl[:,2].mean()])
    
    spine = np.array(spine)
    spine = spine[np.argsort(spine[:,1])]

    # --- Lissage ---
    if do_smooth and len(spine) > smooth_val:
        spine[:,0] = savgol_filter(spine[:,0], smooth_val, 3)
        spine[:,2] = savgol_filter(spine[:,2], smooth_val, 3)

    # --- Calcul des Flèches et Cobb ---
    cobb = compute_cobb_angle(spine[:,0], spine[:,1])
    fd, fl, z_ref = compute_sagittal_arrows(spine)
    dev_front = np.max(np.abs(spine[:,0]))

    # --- Affichage des Métriques ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Angle de Cobb", f"{cobb:.1f}°")
    col2.metric("Flèche Dorsale", f"{fd:.2f} cm")
    col3.metric("Flèche Lombaire", f"{fl:.2f} cm")
    col4.metric("Déviation Lat.", f"{dev_front:.2f} cm")

    # --- Visualisation ---
    tmp = tempfile.gettempdir()
    img_f_path = os.path.join(tmp, "front.png")
    img_s_path = os.path.join(tmp, "side.png")

    fig_f, ax_f = plt.subplots(figsize=(4, 7))
    ax_f.scatter(pts[:,0], pts[:,1], s=1, alpha=0.1, color='gray')
    ax_f.plot(spine[:,0], spine[:,1], 'r-', linewidth=2.5)
    ax_f.set_title("Vue Frontale")
    fig_f.savefig(img_f_path, bbox_inches='tight')

    fig_s, ax_s = plt.subplots(figsize=(4, 7))
    ax_s.scatter(pts[:,2], pts[:,1], s=1, alpha=0.1, color='gray')
    ax_s.plot(spine[:,2], spine[:,1], 'b-', linewidth=2.5, label="Rachis")
    ax_s.plot(z_ref, spine[:,1], 'k--', alpha=0.6, label="Réf. C7-S1")
    ax_s.set_title("Vue Sagittale (Profil)")
    ax_s.legend()
    fig_s.savefig(img_s_path, bbox_inches='tight')

    v1, v2 = st.columns(2)
    v1.pyplot(fig_f)
    v2.pyplot(fig_s)

    # --- Export PDF ---
    res_dict = {"cobb": cobb, "fd": fd, "fl": fl, "dev_f": dev_front}
    patient_dict = {"nom": nom, "prenom": prenom}
    
    pdf_file = export_pdf_pro(patient_dict, res_dict, img_f_path, img_s_path)
    
    with open(pdf_file, "rb") as f:
        st.download_button("📥 Télécharger le Bilan Médical (PDF)", f, f"Bilan_3D_{nom}.pdf", "application/pdf")

else:
    st.info("👋 Bienvenue ! Veuillez charger un fichier .ply pour débuter l'analyse.")
