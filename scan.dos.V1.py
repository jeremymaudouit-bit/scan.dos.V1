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
# IO + PDF
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T

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
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(0.5, 1 * cm))
    story.append(
        Paragraph(
            "<i>Note : La flèche dorsale est la référence (0 cm). La flèche lombaire est mesurée depuis cette verticale dorsale.</i>",
            styles["Italic"],
        )
    )
    story.append(Spacer(1, 1 * cm))

    img_t = Table([[PDFImage(img_f, width=6 * cm, height=9 * cm), PDFImage(img_s, width=6 * cm, height=9 * cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# ROBUST MATH HELPERS
# ==============================
def mad(x):
    x = np.asarray(x)
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def robust_clip_by_mad(x, k=4.0):
    """Return mask keeping values within k*MAD from median."""
    x = np.asarray(x)
    m = np.median(x)
    s = mad(x)
    return np.abs(x - m) <= k * 1.4826 * s

def smooth_spine(spine, w=25, poly=3):
    if spine.shape[0] < 7:
        return spine
    w = int(w)
    if w % 2 == 0:
        w += 1
    # ensure w <= n-1 and odd
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
# AUTO ORIENTATION (TOP FIABILITY)
# ==============================
def detect_vertical_axis(pts):
    """
    Choisit l'axe vertical = celui qui a le plus grand étalement robuste.
    Retourne index 0/1/2.
    """
    spreads = []
    for a in range(3):
        q5, q95 = np.percentile(pts[:, a], [5, 95])
        spreads.append(q95 - q5)
    return int(np.argmax(spreads))

def reorder_axes_to_make_y_vertical(pts, vertical_axis):
    """
    Réordonne pts pour que Y soit vertical.
    Conserve les autres axes dans l'ordre restant.
    """
    axes = [0, 1, 2]
    axes.remove(vertical_axis)
    new_order = [axes[0], vertical_axis, axes[1]]  # X, Y(vertical), Z
    return pts[:, new_order], new_order

def rotate_xz_pca_keep_y(pts):
    """
    Rotation autour de Y via PCA dans le plan XZ.
    Stabilise le frontal/sagittal même si le scan est tourné.
    """
    xz = pts[:, [0, 2]]
    center = np.median(xz, axis=0)
    xz0 = xz - center
    # PCA by SVD
    _, _, Vt = np.linalg.svd(xz0, full_matrices=False)
    R = Vt.T  # 2x2
    xz_rot = xz0 @ R
    out = pts.copy()
    out[:, 0] = xz_rot[:, 0]
    out[:, 2] = xz_rot[:, 1]

    # Make "back" direction positive Z (heuristic)
    if np.percentile(out[:, 2], 90) < 0:
        out[:, 2] *= -1
    # Stabilize X sign
    if np.corrcoef(pts[:, 0], out[:, 0])[0, 1] < 0:
        out[:, 0] *= -1

    return out

# ==============================
# "PERFECT" CENTERLINE: VALLEY (SPINAL GROOVE) + CONTINUITY
# ==============================
def slice_valley_xz(sl, back_percent=80, nbins=60, z_med_smooth_bins=7):
    """
    Axe rachidien ≈ vallée centrale du dos.
    - Prend bande dorsale (z haut) pour isoler le dos
    - Construit profil z(x) (médiane de z par bin x)
    - Lisse légèrement le profil
    - Prend le minimum de z(x) = vallée
    Retourne (x0, z0) ou None
    """
    if sl.shape[0] < 50:
        return None

    # keep dorsal band (back surface), but not only extreme crest
    z_thr = np.percentile(sl[:, 2], back_percent)
    back = sl[sl[:, 2] >= z_thr]
    if back.shape[0] < 30:
        back = sl

    x = back[:, 0]
    z = back[:, 2]

    # robust x-range (avoid arms/outliers)
    xmin, xmax = np.percentile(x, [2, 98])
    if (xmax - xmin) < 1e-6:
        return float(np.median(x)), float(np.median(z))

    edges = np.linspace(xmin, xmax, nbins + 1)
    xc = []
    zprof = []

    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.count_nonzero(m) < 8:
            continue
        xc.append(0.5 * (edges[i] + edges[i + 1]))
        # Use median z for stability (density independent)
        zprof.append(np.median(z[m]))

    if len(zprof) < 10:
        return float(np.median(x)), float(np.median(z))

    xc = np.array(xc, dtype=float)
    zprof = np.array(zprof, dtype=float)

    # light smoothing of profile to avoid bin noise
    w = min(z_med_smooth_bins, len(zprof) - 1)
    if w % 2 == 0:
        w = max(3, w - 1)
    if w >= 3 and len(zprof) > w:
        zprof_s = savgol_filter(zprof, w, 2)
    else:
        zprof_s = zprof

    idx = int(np.argmin(zprof_s))
    return float(xc[idx]), float(zprof[idx])

def build_centerline_valley(pts, n_slices=120, k_mad=4.0,
                            back_percent=80, nbins=60,
                            jump_max_cm=2.0):
    """
    - Tranches en Y
    - Filtre robuste sur X (MAD) dans la tranche
    - Axe par vallée (slice_valley_xz)
    - Enforce continuity (rejette les sauts)
    Retourne spine (Nx3) trié par Y
    """
    y = pts[:, 1]
    y_edges = np.linspace(np.percentile(y, 3), np.percentile(y, 97), n_slices + 1)

    spine = []
    prev_x = None

    for i in range(n_slices):
        sl = pts[(y >= y_edges[i]) & (y < y_edges[i + 1])]
        if sl.shape[0] < 80:
            continue

        # Robust slice cleaning on X & Z
        m1 = robust_clip_by_mad(sl[:, 0], k=k_mad)
        m2 = robust_clip_by_mad(sl[:, 2], k=k_mad)
        sl = sl[m1 & m2]
        if sl.shape[0] < 60:
            continue

        res = slice_valley_xz(sl, back_percent=back_percent, nbins=nbins, z_med_smooth_bins=7)
        if res is None:
            continue
        x0, z0 = res
        y0 = float(np.median(sl[:, 1]))

        # Continuity gate (prevents wild lateral jumps)
        if prev_x is not None and abs(x0 - prev_x) > jump_max_cm:
            # try a more "central" fallback: median x of dorsal band (not perfect but avoids spikes)
            z_thr = np.percentile(sl[:, 2], back_percent)
            back = sl[sl[:, 2] >= z_thr]
            if back.shape[0] >= 30:
                x0_fb = float(np.median(back[:, 0]))
                if abs(x0_fb - prev_x) <= jump_max_cm:
                    x0 = x0_fb
                else:
                    continue
            else:
                continue

        spine.append([x0, y0, z0])
        prev_x = x0

    spine = np.array(spine, dtype=float)
    if spine.shape[0] == 0:
        return spine

    # sort by y
    spine = spine[np.argsort(spine[:, 1])]
    return spine

# ==============================
# METRICS
# ==============================
def compute_sagittal_arrow_lombaire_v2(spine_cm):
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

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    st.subheader("⚙️ Robustesse (recommandé)")
    auto_vertical = st.toggle("Détection auto axe vertical", True)
    auto_align = st.toggle("Auto-alignement rotation (PCA)", True)

    st.divider()
    st.subheader("🧠 Extraction axe (vallée rachidienne)")
    n_slices = st.slider("Nombre de tranches", 60, 200, 140)
    back_percent = st.slider("Bande dorsale (percentile z)", 65, 90, 80)
    nbins = st.slider("Résolution profil (bins X)", 30, 120, 70)
    jump_max = st.slider("Tolérance saut latéral (cm)", 0.5, 5.0, 2.0, 0.1)

    st.divider()
    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- LOAD ---
        pts = load_ply_numpy(ply_file).astype(float) * 0.1  # mm->cm (si déjà cm, retire *0.1)

        # --- AUTO VERTICAL AXIS ---
        if auto_vertical:
            vax = detect_vertical_axis(pts)
            pts, order = reorder_axes_to_make_y_vertical(pts, vax)
        # else: assume y is vertical as in your original pipeline

        # --- CLEAN EXTREMITIES ON Y ---
        y = pts[:, 1]
        mask = (y > np.percentile(y, 5)) & (y < np.percentile(y, 95))
        pts = pts[mask]

        # --- ALIGN IN XZ ---
        if auto_align:
            pts = rotate_xz_pca_keep_y(pts)

        # --- CENTER X ROBUSTLY ---
        pts[:, 0] -= np.median(pts[:, 0])

        # --- BUILD CENTERLINE (VALLEY) ---
        spine = build_centerline_valley(
            pts,
            n_slices=n_slices,
            k_mad=4.0,
            back_percent=back_percent,
            nbins=nbins,
            jump_max_cm=jump_max
        )

        if spine.shape[0] < 12:
            st.error(
                "Extraction insuffisante (pas assez de tranches exploitables). "
                "Essaie : augmenter 'Bande dorsale', augmenter 'Nombre de tranches', ou activer l'auto-alignement."
            )
            st.stop()

        # --- SMOOTH ---
        if do_smooth and spine.shape[0] > smooth_val:
            spine = smooth_spine(spine, w=smooth_val, poly=3)

        # --- METRICS ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # --- PLOTS ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.2)
        ax_f.set_title("Frontale (axe vallée)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.2)
        ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale (axe vallée)", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=160)

        # --- DISPLAY ---
        st.write("### 📈 Analyse Visuelle")
        _, v1, v2, _ = st.columns([1, 1, 1, 1])
        v1.pyplot(fig_f)
        v2.pyplot(fig_s)

        st.write("### 📋 Synthèse des résultats")
        st.markdown(
            f"""
        <div class="result-box">
            <p><b>📏 Flèche Dorsale :</b> <span class="value-text">{fd:.2f} cm</span></p>
            <p><b>📏 Flèche Lombaire :</b> <span class="value-text">{fl:.2f} cm</span></p>
            <p><b>↔️ Déviation Latérale Max :</b> <span class="value-text">{dev_f:.2f} cm</span></p>
            <div class="disclaimer">
                Axe calculé via la <b>vallée rachidienne</b> (profil Z(x) par tranche), auto-alignement PCA (option), rejet des sauts et lissage.
                Cette méthode est beaucoup moins influencée par le nombre de points et suit mieux l'axe central que la moyenne/médiane brute.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # --- EXPORT PDF ---
        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
