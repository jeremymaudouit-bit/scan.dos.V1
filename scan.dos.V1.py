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
    div.stMetric { background-color: #ffffff; border-left: 5px solid #2c3e50; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { background-color: #2c3e50; color: white; height: 3em; border-radius: 8px; font-weight: bold; }
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
    z_ref_line = np.linspace(z[0], z[-1], len(z))
    delta = z - z_ref_line
    # Flèches classiques : Max positif (dorsal) et max négatif (lombaire)
    f_dorsale = abs(np.max(delta))
    f_lombaire = abs(np.min(delta))
    return f_dorsale, f_lombaire, z_ref_line

def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spine_pro.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    
    header_s = ParagraphStyle('Header', fontSize=20, textColor=colors.HexColor("#2c3e50"), spaceAfter=20, alignment=1)
    
    story = []
    story.append(Paragraph("<b>BILAN DE SANTÉ RACHIDIENNE 3D</b>", header_s))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles['Normal']))
    story.append(Paragraph(f"<b>Date de l'analyse :</b> {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    story.append(Spacer(1, 1*cm))

    # Tableau des mesures
    data = [
        ["Indicateur", "Valeur Mesurée", "Statut"],
        ["Angle de Cobb", f"{results['cobb']:.1f}°", "Analyse Frontale"],
        ["Flèche Dorsale (Gîte)", f"{results['fd']:.2f} cm", "Analyse Profil"],
        ["Flèche Lombaire", f"{results['fl']:.2f} cm", "Analyse Profil"],
        ["Déviation Latérale Max", f"{results['dev_f']:.2f} cm", "Symétrie"]
    ]
    t = Table(data, colWidths=[6*cm, 5*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 1*cm))

    # Images
    img_t = Table([[PDFImage(img_f, width=8*cm, height=10*cm), PDFImage(img_s, width=8*cm, height=10*cm)]])
    story.append(img_t)
    
    doc.build(story)
    return path

# ==============================
# INTERFACE SIDEBAR
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()
    st.subheader("🛠 Configuration")
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Précision (fenêtre)", 5, 51, 25, step=2)
    k_std = st.slider("Filtrage points aberrants", 0.5, 3.0, 1.5)
    st.divider()
    ply_file = st.file_uploader("Importer Scan 3D (.PLY)", type=["ply"])
    
    st.info("Le bouton 'Lancer l'analyse' apparaîtra une fois le fichier chargé.")

# ==============================
# MAIN APP
# ==============================
st.title("🦴 SpineScan Pro")

if ply_file:
    # BOUTON ANALYSE (Remis en évidence)
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        with st.spinner("Calcul des paramètres cinématiques..."):
            
            # --- Traitement ---
            pts = load_ply_numpy(ply_file) * 0.1 # cm
            mask = (pts[:,1] > np.percentile(pts[:,1], 5)) & (pts[:,1] < np.percentile(pts[:,1], 95))
            pts = pts[mask]
            pts[:,0] -= pts[:,0].mean() # Centrage horizontal
            
            slices = np.linspace(pts[:,1].min(), pts[:,1].max(), 65)
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

            # --- Lissage (Rétabli) ---
            if do_smooth and len(spine) > smooth_val:
                spine[:,0] = savgol_filter(spine[:,0], smooth_val, 3)
                spine[:,2] = savgol_filter(spine[:,2], smooth_val, 3)

            # --- Calculs ---
            cobb = compute_cobb_angle(spine[:,0], spine[:,1])
            fd, fl, z_ref = compute_sagittal_arrows(spine)
            dev_front = np.max(np.abs(spine[:,0]))

            # --- Affichage Streamlit (Tableau de bord) ---
            st.subheader("📋 Résultats de la synthèse")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Angle de Cobb", f"{cobb:.1f}°")
            c2.metric("Flèche Dorsale", f"{fd:.2f} cm")
            c3.metric("Flèche Lombaire", f"{fl:.2f} cm")
            c4.metric("Déviation Max", f"{dev_front:.2f} cm")

            # Expander pour les valeurs brutes (Optionnel mais Pro)
            with st.expander("🔍 Voir les détails des mesures"):
                st.write(f"**Équilibre frontal :** {dev_front:.2f} cm")
                st.write(f"**Sommet cyphose :** {spine[:,2].max():.2f} cm")
                st.write(f"**Sommet lordose :** {spine[:,2].min():.2f} cm")

            # --- Graphiques ---
            tmp = tempfile.gettempdir()
            img_f_path, img_s_path = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

            fig_f, ax_f = plt.subplots(figsize=(4, 8))
            ax_f.scatter(pts[:,0], pts[:,1], s=0.8, alpha=0.1, color='gray')
            ax_f.plot(spine[:,0], spine[:,1], 'red', linewidth=3, label="Ligne rachidienne")
            ax_f.set_title("PROJECTION FRONTALE", fontweight="bold")
            ax_f.grid(True, linestyle='--', alpha=0.5)
            fig_f.savefig(img_f_path, bbox_inches='tight')

            fig_s, ax_s = plt.subplots(figsize=(4, 8))
            ax_s.scatter(pts[:,2], pts[:,1], s=0.8, alpha=0.1, color='gray')
            ax_s.plot(spine[:,2], spine[:,1], 'blue', linewidth=3, label="Rachis 3D")
            ax_s.plot(z_ref, spine[:,1], 'k--', alpha=0.7, label="Ligne C7-S1")
            ax_s.set_title("PROFIL SAGITTAL", fontweight="bold")
            ax_s.grid(True, linestyle='--', alpha=0.5)
            fig_s.savefig(img_s_path, bbox_inches='tight')

            v1, v2 = st.columns(2)
            v1.pyplot(fig_f)
            v2.pyplot(fig_s)

            # --- Export PDF ---
            res_dict = {"cobb": cobb, "fd": fd, "fl": fl, "dev_f": dev_front}
            pdf_file = export_pdf_pro({"nom": nom, "prenom": prenom}, res_dict, img_f_path, img_s_path)
            
            st.success("✅ Analyse terminée. Le rapport est prêt.")
            with open(pdf_file, "rb") as f:
                st.download_button("📥 TÉLÉCHARGER LE BILAN PROFESSIONNEL (PDF)", f, f"Bilan_{nom}.pdf", "application/pdf")

else:
    st.info("💡 **Instructions :** Chargez un fichier de scan au format .PLY pour activer les outils de diagnostic.")
