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
    Référence sagittale:
      - verticale passant par le point le plus dorsal (max z)
      - flèche dorsale = 0 (référence)
      - flèche lombaire = distance en z entre le point le plus lordotique (min z) et la verticale dorsale
    """
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]

    idx_dorsal = np.argmax(z)
    z_dorsal = z[idx_dorsal]
    vertical_z = np.full_like(y, z_dorsal)

    idx_lombaire = np.argmin(z)
    z_lombaire = z[idx_lombaire]

    fd = 0.0
    fl = float(abs(z_lombaire - z_dorsal))
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
# EXTRACTION ROBUSTE "AXE" (NON INFLUENCÉE PAR LA DENSITÉ)
# ==============================
def robust_spine_centerline(pts_cm, n_slices=70, back_percent=90, x_trim=(25, 75), min_points_slice=80):
    """
    Objectif: extraire une ligne représentative (courbures/forme) sans dépendre du nombre de points.

    Idée:
      1) découper en tranches en y
      2) dans chaque tranche, ne garder que la surface postérieure ("dos") via percentile sur z (top X%)
      3) prendre un centre robuste (médiane), + trimming sur x pour éviter épaules/flancs
      4) renvoyer la polyline (x,y,z) en cm

    Paramètres:
      - back_percent: 90 => on garde les 10% des points les plus "dorsaux" (z élevé)
      - x_trim: (25,75) => on garde le cœur central en x (anti-épaules/flancs), puis médiane
      - min_points_slice: si tranche trop pauvre -> ignorée
    """
    y = pts_cm[:, 1]
    y_edges = np.linspace(y.min(), y.max(), n_slices + 1)

    spine = []
    for i in range(n_slices):
        sl = pts_cm[(y >= y_edges[i]) & (y < y_edges[i+1])]
        if sl.shape[0] < min_points_slice:
            continue

        # 1) garder la surface du dos: points les plus postérieurs (z élevés)
        z_thr = np.percentile(sl[:, 2], back_percent)
        back = sl[sl[:, 2] >= z_thr]
        if back.shape[0] < max(20, min_points_slice // 4):
            continue

        # 2) trimming central sur x pour éviter épaules/flancs (et limiter l'influence densité)
        q1, q2 = np.percentile(back[:, 0], x_trim)
        core = back[(back[:, 0] >= q1) & (back[:, 0] <= q2)]
        if core.shape[0] < 10:
            core = back

        # 3) centre robuste (médiane)
        x0 = float(np.median(core[:, 0]))
        y0 = float(np.median(core[:, 1]))
        z0 = float(np.median(core[:, 2]))
        spine.append([x0, y0, z0])

    spine = np.array(spine, dtype=float)
    return spine

# ==============================
# LOGIQUE PRINCIPALE
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    st.subheader("🧠 Extraction Axe (fiabilité)")
    # Garder le même environnement UI, mais on ajoute juste des réglages robustes
    n_slices = st.slider("Nombre de tranches", 40, 120, 70)
    back_percent = st.slider("Sélection dos (percentile z)", 80, 97, 90)
    xtrim_low = st.slider("Trim x bas (%)", 5, 45, 25)
    xtrim_high = st.slider("Trim x haut (%)", 55, 95, 75)

    st.divider()
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)

    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- CHARGEMENT & PRÉTRAITEMENT ---
        # Conversion mm -> cm (si tes PLY sont en mm). Si c'est déjà en cm, enlève *0.1
        pts = load_ply_numpy(ply_file) * 0.1

        # Nettoyage grossier sur y (retirer extrémités souvent bruitées)
        y = pts[:, 1]
        mask = (y > np.percentile(y, 5)) & (y < np.percentile(y, 95))
        pts = pts[mask]

        # Centrage latéral global (anti-biais si scan décalé)
        pts[:, 0] -= np.median(pts[:, 0])

        # --- EXTRACTION ROBUSTE DE LA LIGNE (peu sensible au nombre de points) ---
        spine = robust_spine_centerline(
            pts_cm=pts,
            n_slices=n_slices,
            back_percent=back_percent,
            x_trim=(xtrim_low, xtrim_high),
            min_points_slice=60
        )

        if spine.shape[0] < 10:
            st.error("Extraction insuffisante: pas assez de points exploitables par tranche. Essaie d'augmenter le nombre de points ou d'ajuster 'Sélection dos'.")
            st.stop()

        # --- LISSAGE (sur une ligne déjà robuste) ---
        if do_smooth:
            # fenêtre impaire, <= longueur-1
            w = int(smooth_val)
            if w % 2 == 0:
                w += 1
            w = min(w, spine.shape[0] - 1 if (spine.shape[0] - 1) % 2 == 1 else spine.shape[0] - 2)
            if w >= 5:
                spine[:, 0] = savgol_filter(spine[:, 0], w, 3)
                spine[:, 2] = savgol_filter(spine[:, 2], w, 3)

        # --- INDICATEURS ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # --- GRAPHES ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color='gray')
        ax_f.plot(spine[:, 0], spine[:, 1], 'red', linewidth=1.8)
        ax_f.set_title("Frontale (axe robuste)", fontsize=9)
        ax_f.axis('off')
        fig_f.savefig(img_f_p, bbox_inches='tight', dpi=150)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color='gray')
        ax_s.plot(spine[:, 2], spine[:, 1], 'blue', linewidth=1.8)
        ax_s.plot(vertical_z, spine[:, 1], 'k--', alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale (axe robuste)", fontsize=9)
        ax_s.axis('off')
        fig_s.savefig(img_s_p, bbox_inches='tight', dpi=150)

        # --- AFFICHAGE ---
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
                Note : l'axe est extrait de manière robuste (médiane + sélection "dos") pour limiter l'influence de la densité de points du scan.
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
