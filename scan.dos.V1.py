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
# ROBUSTESSE "TOP" : ALIGNEMENT + EXTRACTION AXE
# ==============================
def rotate_xz_pca_keep_y(pts_cm):
    """
    Aligne automatiquement le scan dans le plan XZ (rotation autour de Y),
    pour stabiliser le plan frontal (X/Y) et sagittal (Z/Y) même si le patient est légèrement tourné.
    On conserve Y tel quel (vertical).

    Retourne pts_rot, R (2x2), center_xz (2,)
    """
    xz = pts_cm[:, [0, 2]]
    center = np.median(xz, axis=0)
    xz0 = xz - center

    # PCA via SVD (robuste et simple)
    # Vt[0] = direction de variance max ; Vt[1] = variance min
    _, _, Vt = np.linalg.svd(xz0, full_matrices=False)
    R = Vt.T  # 2x2 (colonnes = axes principaux)

    xz_rot = xz0 @ R  # projection
    pts_rot = pts_cm.copy()
    pts_rot[:, 0] = xz_rot[:, 0]
    pts_rot[:, 2] = xz_rot[:, 1]

    # Convention : on veut que Z (profondeur) corresponde à "dos" = valeurs positives
    # Si le dos (top z) est plutôt négatif, on inverse l'axe z
    z = pts_rot[:, 2]
    if np.percentile(z, 90) < 0:
        pts_rot[:, 2] *= -1

    # Convention : on veut X stable (gauche/droite). (Pas indispensable, mais évite flip)
    # On force X à garder le même sens que l'original en corrélation grossière.
    if np.corrcoef(pts_cm[:, 0], pts_rot[:, 0])[0, 1] < 0:
        pts_rot[:, 0] *= -1

    return pts_rot, R, center

def slice_center_x_by_dorsal_profile(sl, back_percent=90, nbins=35, zq=92, min_bin_pts=15):
    """
    Dans une tranche (y), estime X de l'axe via un profil dorsal:
      - garde points les plus dorsaux (top z%)
      - binning en X
      - score bin = quantile haut de Z (zq)
      - choisit bin au score max => X0 (centre du bin)
    Très peu sensible à la densité et beaucoup plus stable en frontal.
    """
    if sl.shape[0] < 30:
        return None

    z_thr = np.percentile(sl[:, 2], back_percent)
    back = sl[sl[:, 2] >= z_thr]
    if back.shape[0] < 20:
        back = sl

    xs = back[:, 0]
    zs = back[:, 2]

    x_min, x_max = np.percentile(xs, [2, 98])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or (x_max - x_min) < 1e-6:
        return float(np.median(xs))

    edges = np.linspace(x_min, x_max, nbins + 1)
    best_score = -np.inf
    best_x = None

    for i in range(nbins):
        m = (xs >= edges[i]) & (xs < edges[i + 1])
        if np.count_nonzero(m) < min_bin_pts:
            continue
        score = np.percentile(zs[m], zq)
        if score > best_score:
            best_score = score
            best_x = 0.5 * (edges[i] + edges[i + 1])

    if best_x is None:
        return float(np.median(xs))
    return float(best_x)

def slice_center_z_by_dorsal_quantile(sl, back_percent=90, zq=92):
    """
    Pour Z (sagittal), on veut suivre la surface postérieure de manière stable.
    On prend directement un quantile haut de Z.
    """
    if sl.shape[0] < 30:
        return None
    # Option: on peut d'abord filtrer "dos" puis prendre quantile
    z_thr = np.percentile(sl[:, 2], back_percent)
    back = sl[sl[:, 2] >= z_thr]
    if back.shape[0] < 20:
        back = sl
    return float(np.percentile(back[:, 2], zq))

def build_spine_centerline_top(pts_cm, n_slices=80, k_std=1.5,
                              back_percent=90, nbins=35, zq=92,
                              min_points_slice=80):
    """
    Axe "top fiabilité":
      - tranche en Y
      - garde ton filtre k_std (anti points aberrants)
      - X0 via profil dorsal (binning) => frontal propre
      - Z0 via quantile dorsal => sagittal propre
      - centres en médiane sur Y
    """
    y = pts_cm[:, 1]
    edges = np.linspace(y.min(), y.max(), n_slices + 1)

    spine = []
    for i in range(n_slices):
        sl = pts_cm[(y >= edges[i]) & (y < edges[i + 1])]
        if sl.shape[0] < min_points_slice:
            continue

        # filtre existant (conserve ce qui marche)
        mx, sx = sl[:, 0].mean(), sl[:, 0].std()
        if sx > 1e-9:
            sl = sl[(sl[:, 0] > mx - k_std * sx) & (sl[:, 0] < mx + k_std * sx)]
        if sl.shape[0] < max(30, min_points_slice // 2):
            continue

        x0 = slice_center_x_by_dorsal_profile(sl, back_percent=back_percent, nbins=nbins, zq=zq, min_bin_pts=max(10, min_points_slice // 10))
        z0 = slice_center_z_by_dorsal_quantile(sl, back_percent=back_percent, zq=zq)
        if x0 is None or z0 is None:
            continue

        y0 = float(np.median(sl[:, 1]))
        spine.append([x0, y0, z0])

    return np.array(spine, dtype=float)

def smooth_spine(spine, smooth_val=25, poly=3):
    if spine.shape[0] < 7:
        return spine
    w = int(smooth_val)
    if w % 2 == 0:
        w += 1
    # fenêtre impaire <= n-1 (et >= 5)
    max_w = spine.shape[0] - 1
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
# LOGIQUE PRINCIPALE
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    # (ce qui fonctionne) lissage + filtre
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)
    k_std = st.slider("Filtre points", 0.5, 3.0, 1.5)

    st.subheader("🧠 Fiabilité (axe colonne)")
    auto_align = st.toggle("Auto-alignement (recommandé)", True)
    n_slices = st.slider("Nombre de tranches", 50, 140, 90)
    back_percent = st.slider("Sélection dos (percentile z)", 80, 97, 90)
    nbins = st.slider("Résolution profil X (bins)", 20, 60, 35)
    zq = st.slider("Quantile dorsal (zq)", 85, 97, 92)

    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- CHARGEMENT ---
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm (si besoin)

        # Nettoyage grossier sur y (comme avant)
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 5)) & (pts[:, 1] < np.percentile(pts[:, 1], 95))
        pts = pts[mask]

        # Auto-align (rotation autour de Y) pour stabiliser frontal/sagittal
        if auto_align:
            pts, _, _ = rotate_xz_pca_keep_y(pts)

        # Centrage latéral robuste (évite un décalage global)
        pts[:, 0] -= np.median(pts[:, 0])

        # --- EXTRACTION AXE "TOP" ---
        spine = build_spine_centerline_top(
            pts_cm=pts,
            n_slices=n_slices,
            k_std=k_std,
            back_percent=back_percent,
            nbins=nbins,
            zq=zq,
            min_points_slice=60
        )

        if spine.shape[0] < 10:
            st.error("Extraction insuffisante. Essaie: augmenter 'Nombre de tranches', diminuer 'Sélection dos' ou activer l'auto-alignement.")
            st.stop()

        # Lissage (après extraction robuste)
        if do_smooth:
            spine = smooth_spine(spine, smooth_val=smooth_val, poly=3)

        # --- INDICATEURS ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # --- GRAPHES ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color='gray')
        ax_f.plot(spine[:, 0], spine[:, 1], 'red', linewidth=1.8)
        ax_f.set_title("Frontale (axe fiable)", fontsize=9)
        ax_f.axis('off')
        fig_f.savefig(img_f_p, bbox_inches='tight', dpi=150)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color='gray')
        ax_s.plot(spine[:, 2], spine[:, 1], 'blue', linewidth=1.8)
        ax_s.plot(vertical_z, spine[:, 1], 'k--', alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale (axe fiable)", fontsize=9)
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
                Axe extrait de façon robuste: auto-alignement (optionnel) + profil dorsal par tranche (peu sensible à la densité).
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
