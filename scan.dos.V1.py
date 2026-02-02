import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
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
    .result-box { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    .value-text { font-size: 1.2rem; font-weight: bold; color: #2c3e50; }
    .stButton>button { background-color: #2c3e50; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
    .disclaimer { font-size: 0.85rem; color: #555; font-style: italic; margin-top: 15px; border-left: 3px solid #ccc; padding-left: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# FONCTIONS TECHNIQUES
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata['vertex']
    return np.vstack([v['x'], v['y'], v['z']]).T

def compute_sagittal_arrow_lombaire(spine_cm):
    """
    Ligne verticale sur la tangente dorsale (z[0] -> z[-1]).
    Flèche dorsale = 0
    Flèche lombaire = distance horizontale (z) de la lordose lombaire à cette verticale
    """
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]

    # Tangente dorsale : ligne entre premier et dernier point
    z_ref_line = np.linspace(z[0], z[-1], len(z))

    # Index du point le plus lordotique (min de z)
    idx_lombaire = np.argmin(z)
    z_lombaire = z[idx_lombaire]

    # Flèche dorsale = 0
    fd = 0.0

    # Flèche lombaire = distance verticale entre lordose et tangente
    fl = abs(z_lombaire - z_ref_line[idx_lombaire])

    return fd, fl, z_ref_line

def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spine_pro.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    header_s = ParagraphStyle('Header', fontSize=18, textColor=colors.HexColor("#2c3e50"), alignment=1)
    
    story = []
    story.append(Paragraph("<b>BILAN DE SANTÉ RACHIDIENNE 3D</b>", header_s))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles['Normal']))
    
    data = [["Indicateur", "Valeur Mesurée"], 
            ["Flèche Dorsale", f"{results['fd']:.2f} cm"],
            ["Flèche Lombaire", f"{results['fl']:.2f} cm"],
            ["Déviation Latérale Max", f"{results['dev_f']:.2f} cm"]]
    
    t = Table(data, colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
                           ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                           ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                           ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.append(t)
    story.append(Spacer(0.5, 1*cm))
    story.append(Paragraph("<i>Note : La flèche dorsale est la référence (0 cm). La flèche lombaire est mesurée à partir de cette verticale dorsale.</i>", styles['Italic']))
    story.append(Spacer(1, 1*cm))
    
    img_t = Table([[PDFImage(img_f, width=6*cm, height=9*cm), PDFImage(img_s, width=6*cm, height=9*cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# LOGIQUE PRINCIPALE
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)
    k_std = st.slider("Filtre points", 0.5, 3.0, 1.5)
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- CALCULS ---
        pts = load_ply_numpy(ply_file) * 0.1 
        mask = (pts[:,1] > np.percentile(pts[:,1], 5)) & (pts[:,1] < np.percentile(pts[:,1], 95))
        pts = pts[mask]
        pts[:,0] -= pts[:,0].mean()
        
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
        if do_smooth and len(spine) > smooth_val:
            spine[:,0] = savgol_filter(spine[:,0], smooth_val, 3)
            spine[:,2] = savgol_filter(spine[:,2], smooth_val, 3)

        # --- FLÈCHES SAGITTALES ---
        fd, fl, z_ref = compute_sagittal_arrow_lombaire(spine)
        dev_f = np.max(np.abs(spine[:,0]))

        # --- GRAPHES (CENTRES) ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:,0], pts[:,1], s=0.2, alpha=0.1, color='gray')
        ax_f.plot(spine[:,0], spine[:,1], 'red', linewidth=1.5)
        ax_f.set_title("Frontale", fontsize=9)
        ax_f.axis('off')
        fig_f.savefig(img_f_p, bbox_inches='tight', dpi=150)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:,2], pts[:,1], s=0.2, alpha=0.1, color='gray')
        ax_s.plot(spine[:,2], spine[:,1], 'blue', linewidth=1.5)
        ax_s.plot(z_ref, spine[:,1], 'k--', alpha=0.5, linewidth=1)
        ax_s.set_title("Sagittale", fontsize=9)
        ax_s.axis('off')
        fig_s.savefig(img_s_p, bbox_inches='tight', dpi=150)

        # --- AFFICHAGE ÉPURÉ ---
        st.write("### 📈 Analyse Visuelle")
        _, v1, v2, _ = st.columns([1, 1, 1, 1])
        v1.pyplot(fig_f)
        v2.pyplot(fig_s)

        st.write("### 📋 Synthèse des résultats")
        st.markdown(f"""
        <div class="result-box">
            <p><b>📏 Flèche Dorsale :</b> <span class="value-text">{fd:.2f} cm</span></p>
            <p><b>📏 Flèche Lombaire :</b> <span class="value-text">{fl:.2f} cm</span></p>
            <p><b>↔️ Déviation Latérale Max :</b> <span class="value-text">{dev_f:.2f} cm</span></p>
            <div class="disclaimer">
                Note : La flèche dorsale est la référence (0 cm). La flèche lombaire est mesurée depuis cette verticale dorsale.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- EXPORT PDF ---
        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)
        
        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
