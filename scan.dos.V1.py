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

def compute_sagittal_arrow_lombaire_v2(spine_cm):
    """
    La verticale de référence passe par le point dorsal le plus haut (max de z)
    Flèche dorsale = 0
    Flèche lombaire = distance horizontale (z) de la lordose lombaire à cette verticale
    """
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]

    idx_dorsal = np.argmax(z)
    z_dorsal = z[idx_dorsal]
    vertical_z = np.full_like(y, z_dorsal)

    idx_lombaire = np.argmin(z)
    z_lombaire = z[idx_lombaire]

    fd = 0.0
    fl = abs(z_lombaire - z_dorsal)

    return fd, fl, vertical_z

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
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(t)
    story.append(Spacer(0.5, 1*cm))
    story.append(Paragraph("<i>Note : La flèche dorsale est la référence (0 cm). La flèche lombaire est mesurée depuis cette verticale dorsale.</i>", styles['Italic']))
    story.append(Spacer(1, 1*cm))

    img_t = Table([[PDFImage(img_f, width=6*cm, height=9*cm), PDFImage(img_s, width=6*cm, height=9*cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# AJOUT MINIMUM : EXTRACTION AXE ROBUSTE (ANTI-DENSITÉ)
# ==============================
def robust_spine_from_slices(pts_cm, n_slices=60, k_std=1.5, back_percent=90, core_percent=40):
    """
    Remplace le 'mean' sensible à la densité par une extraction robuste :
      - tranche en y
      - filtre outliers sur x (comme avant)
      - garde la surface du dos (top z%) pour suivre la colonne / sillon
      - centre par médiane + coeur symétrique autour de la médiane (|x-x_med|)
    """
    y = pts_cm[:, 1]
    edges = np.linspace(y.min(), y.max(), n_slices + 1)

    spine = []
    for i in range(n_slices):
        sl = pts_cm[(y >= edges[i]) & (y < edges[i+1])]
        if sl.shape[0] <= 5:
            continue

        # --- garde ton filtre "k_std" (ce qui fonctionne) ---
        mx, sx = sl[:, 0].mean(), sl[:, 0].std()
        if sx > 1e-9:
            sl = sl[(sl[:, 0] > mx - k_std*sx) & (sl[:, 0] < mx + k_std*sx)]
        if sl.shape[0] < 10:
            continue

        # --- AJOUT: ne garder que le dos (points les plus postérieurs) ---
        z_thr = np.percentile(sl[:, 2], back_percent)
        back = sl[sl[:, 2] >= z_thr]
        if back.shape[0] < 10:
            back = sl  # fallback

        # --- AJOUT: centre robuste (médiane) + coeur symétrique (anti dé-axage) ---
        x_med = np.median(back[:, 0])
        dx = np.abs(back[:, 0] - x_med)
        dx_thr = np.percentile(dx, core_percent)
        core = back[dx <= dx_thr]
        if core.shape[0] < 5:
            core = back

        spine.append([np.median(core[:, 0]), np.median(core[:, 1]), np.median(core[:, 2])])

    return np.array(spine, dtype=float)

# ==============================
# LOGIQUE PRINCIPALE
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    # on garde tes réglages existants
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)
    k_std = st.slider("Filtre points", 0.5, 3.0, 1.5)

    # AJOUT MINIMUM: 2 réglages robustes (optionnels)
    st.subheader("🧠 Axe robuste (anti-densité)")
    back_percent = st.slider("Dos (percentile z)", 80, 97, 90)
    core_percent = st.slider("Cœur central (%)", 10, 70, 40)

    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- CALCULS ---
        pts = load_ply_numpy(ply_file) * 0.1
        mask = (pts[:,1] > np.percentile(pts[:,1], 5)) & (pts[:,1] < np.percentile(pts[:,1], 95))
        pts = pts[mask]

        # garde ton centrage global (mais en médiane = plus robuste, sans casser le reste)
        pts[:,0] -= np.median(pts[:,0])

        # --- CHANGEMENT MINIMUM ICI : extraction spine robuste au lieu du mean simple ---
        slices = 60  # identique à ton code d'origine
        spine = robust_spine_from_slices(
            pts_cm=pts,
            n_slices=slices,
            k_std=k_std,
            back_percent=back_percent,
            core_percent=core_percent
        )

        if spine.shape[0] < 8:
            st.error("Pas assez de tranches exploitables. Essaie d'augmenter 'Dos (percentile z)' ou 'Filtre points'.")
            st.stop()

        if do_smooth and len(spine) > smooth_val:
            w = int(smooth_val)
            if w % 2 == 0:
                w += 1
            w = min(w, len(spine)-1 if (len(spine)-1) % 2 == 1 else len(spine)-2)
            if w >= 5:
                spine[:,0] = savgol_filter(spine[:,0], w, 3)
                spine[:,2] = savgol_filter(spine[:,2], w, 3)

        # --- FLÈCHES SAGITTALES ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
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
        ax_s.plot(vertical_z, spine[:,1], 'k--', alpha=0.7, linewidth=1)
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
                Note : extraction robuste (médiane + dos percentile z + cœur symétrique) pour limiter l'influence du nombre de points du scan.
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
