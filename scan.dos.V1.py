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
    .result-box { background-color:#fff; padding:14px; border-radius:10px; border:1px solid #e0e0e0; margin-bottom:10px; }
    .value-text { font-size: 1.1rem; font-weight: bold; color: #2c3e50; }
    .stButton>button { background-color: #2c3e50; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
    .disclaimer { font-size: 0.82rem; color: #555; font-style: italic; margin-top: 10px; border-left: 3px solid #ccc; padding-left: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# IO
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

# ==============================
# PDF
# ==============================
def export_pdf_pro(patient_info, results, img_f, img_s):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_spine_pro.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    header_s = ParagraphStyle("Header", fontSize=16, textColor=colors.HexColor("#2c3e50"), alignment=1)

    story = []
    story.append(Paragraph("<b>BILAN DE SANTÉ RACHIDIENNE 3D</b>", header_s))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

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
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    img_t = Table([[PDFImage(img_f, width=6.2 * cm, height=9.0 * cm),
                    PDFImage(img_s, width=6.2 * cm, height=9.0 * cm)]])
    story.append(img_t)
    doc.build(story)
    return path

# ==============================
# METRICS (sagittal)
# ==============================
def compute_sagittal_arrow_lombaire_v2(spine_cm):
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]
    if len(z) == 0:
        return 0.0, 0.0, np.array([])
    idx_dorsal = int(np.argmax(z))
    z_dorsal = float(z[idx_dorsal])
    vertical_z = np.full_like(y, z_dorsal)
    idx_lombaire = int(np.argmin(z))
    z_lombaire = float(z[idx_lombaire])
    fd = 0.0
    fl = float(abs(z_lombaire - z_dorsal))
    return fd, fl, vertical_z

# ==============================
# LISSAGE
# ==============================
def median_filter_1d(a, k):
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0:
        return a
    k = int(k)
    if k < 3:
        return a
    if k % 2 == 0:
        k += 1
    r = k // 2
    out = np.empty_like(a)
    for i in range(n):
        lo = max(0, i - r)
        hi = min(n, i + r + 1)
        out[i] = np.median(a[lo:hi])
    return out

def smooth_spine(spine, window=91, strong=True, median_k=11):
    if spine.shape[0] < 7:
        return spine
    out = spine.copy()
    n = out.shape[0]

    if strong:
        mk = int(median_k)
        if mk % 2 == 0:
            mk += 1
        mk = min(mk, n if n % 2 == 1 else n - 1)
        mk = max(3, mk)
        out[:, 0] = median_filter_1d(out[:, 0], mk)
        out[:, 2] = median_filter_1d(out[:, 2], mk)

    w = int(window)
    if w % 2 == 0:
        w += 1
    max_w = n - 1
    if max_w % 2 == 0:
        max_w -= 1
    w = min(w, max_w)
    if w < 5:
        return out

    out[:, 0] = savgol_filter(out[:, 0], w, 3)
    out[:, 2] = savgol_filter(out[:, 2], w, 3)
    return out

# ==============================
# ROTATION CORRECTION
# ==============================
def estimate_rotation_xz(pts):
    y = pts[:, 1]
    mid = (y > np.percentile(y, 30)) & (y < np.percentile(y, 70))
    pts_mid = pts[mid] if np.count_nonzero(mid) > 200 else pts
    XZ = pts_mid[:, [0, 2]]
    XZ = XZ - np.mean(XZ, axis=0)
    _, _, Vt = np.linalg.svd(XZ, full_matrices=False)
    angle = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
    c, s = np.cos(-angle), np.sin(-angle)
    return np.array([[c, -s], [s, c]], dtype=float)

def apply_rotation_xz(pts, R):
    XZ_rot = pts[:, [0, 2]] @ R.T
    return np.column_stack([XZ_rot[:, 0], pts[:, 1], XZ_rot[:, 1]])

# ==============================
# LIGNE MÉDIANE DOS (density-free)
# ==============================
def slice_profile_surface_binned(sl, cell_cm=0.5, z_percentile=95, min_pts_per_bin=3):
    """
    Profil surface z(x) en bins X :
    - 1 bin = 1 vote (anti biais densité)
    - z dans bin = percentile haut (surface)
    """
    x = sl[:, 0]
    z = sl[:, 2]
    xmin, xmax = np.percentile(x, [2, 98])
    if xmax - xmin < 1e-6:
        return None, None

    nbins = max(35, int(np.ceil((xmax - xmin) / cell_cm)))
    edges = np.linspace(xmin, xmax, nbins + 1)

    xc, zc = [], []
    for b in range(nbins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if np.count_nonzero(m) < min_pts_per_bin:
            continue
        xc.append(0.5 * (edges[b] + edges[b + 1]))
        zc.append(float(np.percentile(z[m], z_percentile)))

    if len(xc) < 12:
        return None, None

    xc = np.array(xc, dtype=float)
    zc = np.array(zc, dtype=float)
    o = np.argsort(xc)
    return xc[o], zc[o]

def midline_from_profile(xc, zc, q_edge=10):
    """
    Milieu entre bords (sur bins), pas une moyenne de points.
    """
    xl = float(np.percentile(xc, q_edge))
    xr = float(np.percentile(xc, 100 - q_edge))
    x_mid = 0.5 * (xl + xr)
    z_mid = float(np.interp(x_mid, xc, zc))
    return x_mid, z_mid

def extract_spine_midline_back(
    pts,
    remove_lateral_outliers=True,
    cell_cm=0.45,
    z_percentile=95,
    q_edge=10,
    y_low=10,
    y_high=92,
    n_slices_min=140,
    n_slices_max=240,
):
    """
    Extraction "ligne médiane dos" :
    - rotation XZ
    - tranches Y
    - profil surface z(x) binned (density-free)
    - bords sur xc -> x_mid
    - z_mid = z(x_mid) via interpolation profil
    """
    R = estimate_rotation_xz(pts)
    pr = apply_rotation_xz(pts, R)

    y = pr[:, 1]
    y0 = np.percentile(y, y_low)
    y1 = np.percentile(y, y_high)

    n_slices = int(np.clip((pr.shape[0] // 3000) + 160, n_slices_min, n_slices_max))
    edges_y = np.linspace(y0, y1, n_slices)

    spine = []
    prev_x = None

    for i in range(len(edges_y) - 1):
        sl = pr[(y >= edges_y[i]) & (y < edges_y[i + 1])]
        if sl.shape[0] < 60:
            continue

        # retire extrêmes latéraux (bras/artefacts), sans utiliser de moyenne
        if remove_lateral_outliers:
            x = sl[:, 0]
            xl, xr = np.percentile(x, [2, 98])
            sl = sl[(x >= xl) & (x <= xr)]
            if sl.shape[0] < 40:
                continue

        xc, zc = slice_profile_surface_binned(
            sl, cell_cm=cell_cm, z_percentile=z_percentile, min_pts_per_bin=3
        )
        if xc is None:
            continue

        # lissage léger du profil (sur bins)
        if len(zc) >= 11:
            zc_s = savgol_filter(zc, 11, 2)
        else:
            zc_s = zc

        x0, z0 = midline_from_profile(xc, zc_s, q_edge=q_edge)
        y_mid = float(np.median(sl[:, 1]))  # médiane, pas moyenne

        # continuité douce (pas trop tôt)
        if prev_x is not None and len(spine) > 12 and abs(x0 - prev_x) > 2.5:
            x0 = prev_x
            # z0 recalculé au centre imposé
            z0 = float(np.interp(x0, xc, zc_s))

        prev_x = x0
        spine.append([x0, y_mid, z0])

    if len(spine) == 0:
        return np.empty((0, 3), dtype=float)

    spine = np.array(spine, dtype=float)
    spine = spine[np.argsort(spine[:, 1])]

    # retour repère original
    XZ_back = spine[:, [0, 2]] @ R
    spine[:, 0] = XZ_back[:, 0]
    spine[:, 2] = XZ_back[:, 1]
    return spine

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    st.subheader("🧭 Ligne médiane dos")
    remove_lat = st.toggle("Limiter artefacts latéraux (bras)", True)
    cell_cm = st.slider("Taille bin X (cm)", 0.25, 1.00, 0.45, step=0.05)
    z_perc = st.slider("Surface dos (percentile Z)", 85, 99, 95, step=1)
    q_edge = st.slider("Bords (quantile X)", 5, 20, 10, step=1)

    st.divider()
    st.subheader("🧽 Lissage")
    do_smooth = st.toggle("Activer", True)
    strong_smooth = st.toggle("Lissage fort (anti-pics)", True)
    smooth_window = st.slider("Fenêtre lissage", 5, 151, 91, step=2)
    median_k = st.slider("Anti-pics (médian)", 3, 31, 11, step=2)

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro — Ligne médiane dos")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm

        # nettoyage Y
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 5)) & (pts[:, 1] < np.percentile(pts[:, 1], 95))
        pts = pts[mask]

        # centrage global X (affichage uniquement, OK)
        pts[:, 0] -= np.median(pts[:, 0])

        # extraction ligne médiane dos (density-free)
        spine = extract_spine_midline_back(
            pts,
            remove_lateral_outliers=remove_lat,
            cell_cm=float(cell_cm),
            z_percentile=float(z_perc),
            q_edge=int(q_edge),
        )

        # fallback soft: relâche artefacts latéraux
        if spine.shape[0] < 10 and remove_lat:
            spine = extract_spine_midline_back(
                pts,
                remove_lateral_outliers=False,
                cell_cm=float(cell_cm),
                z_percentile=float(z_perc),
                q_edge=int(q_edge),
            )

        if spine.shape[0] == 0:
            st.error("Impossible d'extraire une courbe (scan trop incomplet).")
            st.stop()

        # lissage
        if do_smooth:
            spine = smooth_spine(spine, window=smooth_window, strong=strong_smooth, median_k=median_k)

        # métriques
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0]))) if spine.size else 0.0

        # images
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.6)
        ax_f.set_title("Frontale (ligne médiane dos)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.6)
        if vertical_z.size:
            ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=160)

        st.write("### 📈 Analyse Visuelle")
        _, c1, c2, _ = st.columns([1, 1, 1, 1])
        c1.pyplot(fig_f)
        c2.pyplot(fig_s)

        st.write("### 📋 Synthèse des résultats")
        st.markdown(f"""
        <div class="result-box">
            <p><b>📏 Flèche Dorsale :</b> <span class="value-text">{fd:.2f} cm</span></p>
            <p><b>📏 Flèche Lombaire :</b> <span class="value-text">{fl:.2f} cm</span></p>
            <p><b>↔️ Déviation Latérale Max :</b> <span class="value-text">{dev_f:.2f} cm</span></p>
            <div class="disclaimer">
                Ligne médiane dos = milieu entre bords (bins X, anti-biais densité) + z(x_mid) interpolé sur profil surface (Z p{int(z_perc)}).
            </div>
        </div>
        """, unsafe_allow_html=True)

        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
