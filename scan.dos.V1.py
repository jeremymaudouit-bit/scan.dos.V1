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
# IO / PDF
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

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
# HELPERS
# ==============================
def compute_sagittal_arrow_lombaire_v2(spine_cm):
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

def detect_vertical_axis(pts):
    spans = []
    for a in range(3):
        q5, q95 = np.percentile(pts[:, a], [5, 95])
        spans.append(q95 - q5)
    return int(np.argmax(spans))

def reorder_axes_to_make_y_vertical(pts, vertical_axis):
    axes = [0, 1, 2]
    axes.remove(vertical_axis)
    order = [axes[0], vertical_axis, axes[1]]  # X, Y(vertical), Z
    return pts[:, order], order

def infer_scale_to_cm(pts):
    """
    Heuristique d'échelle :
    - Si hauteur (amplitude axe vertical) ~ 1.5 à 2.2 => probablement mètres -> *100
    - Si ~ 150 à 220 => probablement cm -> *1
    - Si ~ 1500 à 2200 => probablement mm -> *0.1
    """
    vax = detect_vertical_axis(pts)
    q5, q95 = np.percentile(pts[:, vax], [5, 95])
    height = float(q95 - q5)

    # bornes larges pour ne pas casser
    if 0.5 < height < 3.0:
        return 100.0, "m → cm"
    if 50.0 < height < 300.0:
        return 1.0, "cm → cm"
    if 500.0 < height < 3000.0:
        return 0.1, "mm → cm"
    # fallback : ne change rien
    return 1.0, "échelle inconnue (×1)"

def voxel_downsample_median(pts_cm, voxel_cm):
    if pts_cm.shape[0] == 0:
        return pts_cm
    q = np.floor(pts_cm / voxel_cm).astype(np.int64)
    key = q[:, 0] * 73856093 + q[:, 1] * 19349663 + q[:, 2] * 83492791
    order = np.argsort(key)
    pts_sorted = pts_cm[order]
    key_sorted = key[order]
    boundaries = np.where(np.diff(key_sorted) != 0)[0] + 1
    groups = np.split(pts_sorted, boundaries)
    out = np.array([np.median(g, axis=0) for g in groups], dtype=float)
    return out

def profile_z_of_x(points, nbins=70):
    if points.shape[0] < 20:
        return None
    x = points[:, 0]
    z = points[:, 2]
    xmin, xmax = np.percentile(x, [2, 98])
    if xmax - xmin < 1e-6:
        return np.array([np.median(x)]), np.array([np.median(z)])
    edges = np.linspace(xmin, xmax, nbins + 1)
    xc, zc = [], []
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if np.count_nonzero(m) < 4:
            continue
        xc.append(0.5 * (edges[i] + edges[i + 1]))
        zc.append(float(np.median(z[m])))
    if len(xc) < 6:
        return None
    xc = np.array(xc, dtype=float)
    zc = np.array(zc, dtype=float)
    w = min(7, len(zc) - 1)
    if w % 2 == 0:
        w -= 1
    if w >= 5 and len(zc) > w:
        zc = savgol_filter(zc, w, 2)
    return xc, zc

def symmetry_center_from_profile(xc, zc, n_candidates=21, trim=0.10):
    if xc.shape[0] < 6:
        return float(np.median(xc))
    xmin, xmax = float(xc.min()), float(xc.max())
    span = xmax - xmin
    a = xmin + trim * span
    b = xmax - trim * span
    if b <= a:
        return float(np.median(xc))
    candidates = np.linspace(a, b, n_candidates)
    best_c, best_cost = None, np.inf
    for c in candidates:
        umax = min(c - xmin, xmax - c)
        if umax <= 0:
            continue
        us = np.linspace(0, umax, 21)
        zL = np.interp(c - us, xc, zc)
        zR = np.interp(c + us, xc, zc)
        cost = float(np.mean(np.abs(zR - zL)))
        if cost < best_cost:
            best_cost, best_c = cost, c
    return float(best_c if best_c is not None else np.median(xc))

def build_spine_axis_symmetry(pts_cm, n_slices=90, back_percent=85, nbins=70):
    """
    Version qui NE BLOQUE PAS :
    - si profil insuffisant => fallback médiane x sur bande dorsale (au lieu d'abandonner la tranche)
    """
    y = pts_cm[:, 1]
    y_min, y_max = np.percentile(y, [5, 95])
    edges = np.linspace(y_min, y_max, n_slices + 1)

    spine = []
    kept = 0
    for i in range(n_slices):
        sl = pts_cm[(y >= edges[i]) & (y < edges[i + 1])]
        if sl.shape[0] < 15:
            continue

        z_thr = np.percentile(sl[:, 2], back_percent)
        back = sl[sl[:, 2] >= z_thr]
        if back.shape[0] < 10:
            back = sl

        prof = profile_z_of_x(back, nbins=nbins)
        if prof is None:
            # fallback (important : ne pas perdre la tranche)
            x0 = float(np.median(back[:, 0]))
        else:
            xc, zc = prof
            x0 = symmetry_center_from_profile(xc, zc)

        y0 = float(np.median(sl[:, 1]))
        z0 = float(np.percentile(back[:, 2], 90))
        spine.append([x0, y0, z0])
        kept += 1

    spine = np.array(spine, dtype=float)
    if spine.shape[0] == 0:
        return spine, kept
    spine = spine[np.argsort(spine[:, 1])]
    return spine, kept

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    do_smooth = st.toggle("Lissage des courbes", True)
    smooth_val = st.slider("Intensité lissage", 5, 51, 25, step=2)

    st.subheader("🧩 Surface-weight (anti densité)")
    voxel_mm = st.slider("Résolution surface (mm)", 1.0, 8.0, 3.0, 0.5)

    st.subheader("🧠 Extraction")
    auto_vertical = st.toggle("Auto-détection axe vertical", True)
    n_slices = st.slider("Nombre de tranches", 50, 200, 120)
    back_percent = st.slider("Bande dorsale (percentile z)", 70, 97, 85)
    nbins = st.slider("Bins profil z(x)", 20, 140, 70)

    st.subheader("🛠 Debug")
    show_debug = st.toggle("Afficher debug", True)

    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        pts_raw = load_ply_numpy(ply_file)

        # 1) Echelle -> cm (auto)
        scale, scale_label = infer_scale_to_cm(pts_raw)
        pts = pts_raw * scale

        # 2) Axe vertical auto
        if auto_vertical:
            vax = detect_vertical_axis(pts)
            pts, order = reorder_axes_to_make_y_vertical(pts, vax)

        # 3) Nettoyage Y
        y = pts[:, 1]
        mask = (y > np.percentile(y, 5)) & (y < np.percentile(y, 95))
        pts = pts[mask]

        # 4) Centrage X robuste
        pts[:, 0] -= np.median(pts[:, 0])

        # 5) Voxel adaptatif si trop agressif
        voxel_cm = float(voxel_mm) / 10.0
        pts_u = voxel_downsample_median(pts, voxel_cm)

        # si on a trop détruit le nuage, on réduit automatiquement le voxel
        # (ça arrive pile dans ton cas)
        if pts_u.shape[0] < 0.08 * pts.shape[0]:
            pts_u = voxel_downsample_median(pts, voxel_cm * 0.5)
            voxel_cm = voxel_cm * 0.5

        # 6) Extraction
        spine, kept = build_spine_axis_symmetry(
            pts_cm=pts_u,
            n_slices=n_slices,
            back_percent=back_percent,
            nbins=nbins
        )

        if show_debug:
            st.info(
                f"Échelle détectée: {scale_label} (×{scale:g}) | "
                f"Points: brut={pts_raw.shape[0]} / nettoyé={pts.shape[0]} / voxel={pts_u.shape[0]} (voxel={voxel_cm*10:.1f} mm) | "
                f"Tranches gardées: {kept}/{n_slices}"
            )

        if spine.shape[0] < 10:
            st.error(
                "Toujours insuffisant. Dans ce cas, la cause est presque toujours : "
                "(1) scan incomplet du dos, (2) axes très différents, ou (3) échelle extrême. "
                "Regarde le debug (points/tranches)."
            )
            st.stop()

        # 7) Lissage
        if do_smooth and spine.shape[0] > smooth_val:
            spine = smooth_spine(spine, smooth_val=smooth_val, poly=3)

        # 8) Metrics
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # 9) Plots
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts_u[:, 0], pts_u[:, 1], s=0.6, alpha=0.10, color="gray")
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.0)
        ax_f.set_title("Frontale (surface+symétrie)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts_u[:, 2], pts_u[:, 1], s=0.6, alpha=0.10, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.0)
        ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=160)

        # 10) Display
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
                Méthode robuste : échelle auto → densité uniformisée (voxel) → axe frontal par symétrie du profil z(x) avec fallback (ne bloque pas).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 11) PDF
        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)
        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
