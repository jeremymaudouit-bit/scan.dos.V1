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
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

def compute_sagittal_arrow_lombaire_v2(spine_cm):
    """
    Verticale de référence : passe par le point le plus dorsal (max z) de la ligne.
    Flèche dorsale = 0 (référence)
    Flèche lombaire = distance en z entre le point le plus lordotique (min z) et la verticale dorsale
    """
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]

    idx_dorsal = int(np.argmax(z))
    z_dorsal = float(z[idx_dorsal])
    vertical_z = np.full_like(y, z_dorsal)

    idx_lombaire = int(np.argmin(z))
    z_lombaire = float(z[idx_lombaire])

    fd = 0.0
    fl = float(abs(z_lombaire - z_dorsal))
    return fd, fl, vertical_z

def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spine_pro.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    header_s = ParagraphStyle("Header", fontSize=18, textColor=colors.HexColor("#2c3e50"), alignment=1)

    story = []
    story.append(Paragraph("<b>BILAN DE SANTÉ RACHIDIENNE 3D</b>", header_s))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles["Normal"]))

    data = [
        ["Indicateur", "Valeur Mesurée"],
        ["Flèche Dorsale", f"{results['fd']:.2f} cm"],
        ["Flèche Lombaire", f"{results['fl']:.2f} cm"],
        ["Déviation Latérale Max", f"{results['dev_f']:.2f} cm"],
    ]

    t = Table(data, colWidths=[7 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(0.5, 1 * cm))
    story.append(Paragraph(
        "<i>Note : La flèche dorsale est la référence (0 cm). La flèche lombaire est mesurée depuis cette verticale dorsale.</i>",
        styles["Italic"]
    ))
    story.append(Spacer(1, 1 * cm))

    img_t = Table([[PDFImage(img_f, width=6 * cm, height=9 * cm), PDFImage(img_s, width=6 * cm, height=9 * cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# NOUVEAU : "MOYENNE DE SURFACE" (anti densité) via voxelisation
# ==============================
def voxel_downsample_median(pts_cm, voxel=0.25):
    """
    Rend la densité quasi uniforme => chaque zone de surface pèse pareil.
    voxel en cm. Ex: 0.25 cm = 2.5 mm.
    Regroupe par cellule 3D puis remplace par médiane (x,y,z).
    """
    if pts_cm.shape[0] == 0:
        return pts_cm

    q = np.floor(pts_cm / voxel).astype(np.int64)

    # hash voxel
    key = q[:, 0] * 73856093 + q[:, 1] * 19349663 + q[:, 2] * 83492791
    order = np.argsort(key)

    pts_sorted = pts_cm[order]
    key_sorted = key[order]

    boundaries = np.where(np.diff(key_sorted) != 0)[0] + 1
    groups = np.split(pts_sorted, boundaries)

    out = np.array([np.median(g, axis=0) for g in groups], dtype=float)
    return out

def smooth_spine(spine, smooth_val=25, poly=3):
    if spine.shape[0] < 7:
        return spine
    w = int(smooth_val)
    if w % 2 == 0:
        w += 1
    n = spine.shape[0]
    max_w = n - 1
    if max_w % 2 == 0:
        max_w -= 1
    w = min(w, max_w)
    if w < 5:
        return spine
    out = spine.copy()
    out[:, 0] = savgol_filter(out[:, 0], w, poly)
    out[:, 2] = savgol_filter(out[:, 2], w, poly)
    return out

# ==============================
# EXTRACTION AXE STABLE (après "surface-weight")
# ==============================
def build_spine_axis_surface_weighted(pts_cm, n_slices=80, back_percent=85, k_std=1.5):
    """
    Axe stable et propre :
    1) pts déjà "surface-weight" (voxelisé)
    2) tranches en y
    3) garder bande dorsale (z haut) pour isoler le dos
    4) AXE = médiane de X sur cette bande (ligne de symétrie du dos)
       (et Z = quantile dorsal pour la vue sagittale stable)
    """
    y = pts_cm[:, 1]
    edges = np.linspace(y.min(), y.max(), n_slices + 1)

    spine = []
    for i in range(n_slices):
        sl = pts_cm[(y >= edges[i]) & (y < edges[i + 1])]
        if sl.shape[0] < 30:
            continue

        # garder ton filtre points (mais moins agressif, car voxelisation = déjà propre)
        mx, sx = sl[:, 0].mean(), sl[:, 0].std()
        if sx > 1e-9:
            sl = sl[(sl[:, 0] > mx - k_std * sx) & (sl[:, 0] < mx + k_std * sx)]
        if sl.shape[0] < 20:
            continue

        # bande dorsale
        z_thr = np.percentile(sl[:, 2], back_percent)
        back = sl[sl[:, 2] >= z_thr]
        if back.shape[0] < 10:
            back = sl

        # axe par symétrie (robuste, frontal propre)
        x0 = float(np.median(back[:, 0]))
        y0 = float(np.median(back[:, 1]))

        # sagittal stable : suivre le dos (quantile haut)
        z0 = float(np.percentile(back[:, 2], 90))

        spine.append([x0, y0, z0])

    spine = np.array(spine, dtype=float)
    if spine.shape[0] == 0:
        return spine
    spine = spine[np.argsort(spine[:, 1])]
    return spine

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    # ce qui fonctionnait déjà
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)
    k_std = st.slider("Filtre points", 0.5, 3.0, 1.2)

    # nouveau : moyenne de surface
    st.subheader("🧩 Moyenne de surface (anti densité)")
    voxel_mm = st.slider("Résolution surface (mm)", 1.5, 6.0, 2.5, 0.5)
    back_percent = st.slider("Bande dorsale (percentile z)", 70, 95, 85)
    n_slices = st.slider("Nombre de tranches", 50, 140, 80)

    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- LOAD ---
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm (si besoin)

        # --- nettoyage y (comme avant) ---
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 5)) & (pts[:, 1] < np.percentile(pts[:, 1], 95))
        pts = pts[mask]

        # --- centrage global X (robuste) ---
        pts[:, 0] -= np.median(pts[:, 0])

        # ✅ NOUVEAU : uniformisation densité (surface-weight)
        voxel_cm = float(voxel_mm) / 10.0
        pts_u = voxel_downsample_median(pts, voxel=voxel_cm)

        # --- extraction axe ---
        spine = build_spine_axis_surface_weighted(
            pts_cm=pts_u,
            n_slices=n_slices,
            back_percent=back_percent,
            k_std=k_std
        )

        if spine.shape[0] < 10:
            st.error("Extraction insuffisante. Essaie d'augmenter le nombre de tranches, ou diminuer la résolution (mm).")
            st.stop()

        # --- lissage ---
        if do_smooth and spine.shape[0] > smooth_val:
            spine = smooth_spine(spine, smooth_val=smooth_val, poly=3)

        # --- métriques ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # --- graphes ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts_u[:, 0], pts_u[:, 1], s=0.6, alpha=0.10, color="gray")
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.0)
        ax_f.set_title("Frontale (surface-weight)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts_u[:, 2], pts_u[:, 1], s=0.6, alpha=0.10, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.0)
        ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale (surface-weight)", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=160)

        # --- affichage ---
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
                Axe calculé après <b>moyenne de surface</b> (voxelisation) pour éviter l'influence du nombre de points.
                Ensuite, extraction par <b>symétrie</b> du dos (médiane X de la bande dorsale) + lissage.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- export PDF ---
        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
