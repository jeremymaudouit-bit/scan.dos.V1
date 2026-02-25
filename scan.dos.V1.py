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
# ROTATION CORRECTION (inchangée)
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
# SYMMETRY (garde ton idée, mais utilisée sur bins)
# ==============================
def symmetry_center_1d(xc, zc):
    if len(xc) < 10:
        return float(np.median(xc))
    xmin, xmax = float(np.min(xc)), float(np.max(xc))
    span = xmax - xmin
    a = xmin + 0.12 * span
    b = xmax - 0.12 * span
    if b <= a:
        return float(np.median(xc))

    candidates = np.linspace(a, b, 31)
    best_c, best_cost = None, np.inf
    for c in candidates:
        umax = min(c - xmin, xmax - c)
        if umax <= 0:
            continue
        us = np.linspace(0, umax, 30)
        zL = np.interp(c - us, xc, zc)
        zR = np.interp(c + us, xc, zc)
        cost = float(np.mean(np.abs(zR - zL)))
        if cost < best_cost:
            best_cost, best_c = cost, c
    return float(best_c if best_c is not None else np.median(xc))

# ==============================
# DENSITY-FREE PROFILE (anti biais densité)
# ==============================
def slice_profile_surface_binned(sl, cell_cm=0.5, z_percentile=95):
    """
    Crée un profil z(x) en bins réguliers en X.
    Chaque bin compte 1 fois -> pas de biais densité.
    On prend un percentile haut de Z dans chaque bin (surface du dos).
    """
    x = sl[:, 0]
    z = sl[:, 2]

    xmin, xmax = np.percentile(x, [2, 98])
    if xmax - xmin < 1e-6:
        return None, None

    nbins = max(30, int(np.ceil((xmax - xmin) / cell_cm)))
    edges = np.linspace(xmin, xmax, nbins + 1)

    xc, zc = [], []
    for b in range(nbins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if np.count_nonzero(m) < 3:
            continue
        xc.append(0.5 * (edges[b] + edges[b + 1]))
        zc.append(float(np.percentile(z[m], z_percentile)))

    if len(xc) < 12:
        return None, None

    xc = np.array(xc, dtype=float)
    zc = np.array(zc, dtype=float)
    o = np.argsort(xc)
    return xc[o], zc[o]

def robust_center_from_bins(xc):
    """
    Centre robuste "density-free" : médiane des centres de bins valides.
    (chaque bin = 1 vote)
    """
    if xc is None or len(xc) == 0:
        return None
    return float(np.median(xc))

# ==============================
# EXTRACTION CORRIGÉE (sans biais densité)
# ==============================
def extract_spine_density_free(pts, remove_shoulders=True):
    """
    Extraction robuste + anti-biais densité :
    - rotation XZ
    - slices Y
    - profil z(x) en bins X (density-free)
    - centre par symétrie du profil (ou médiane bins)
    - Z = percentile haut de la tranche
    """
    R = estimate_rotation_xz(pts)
    pts_r = apply_rotation_xz(pts, R)

    y = pts_r[:, 1]
    y0 = np.percentile(y, 10)
    y1 = np.percentile(y, 92)

    n_slices = int(np.clip((pts_r.shape[0] // 3000) + 140, 140, 240))
    slices = np.linspace(y0, y1, n_slices)

    spine = []
    prev_x = None

    # bins X : adapte selon densité globale, mais reste borné
    cell_cm = float(np.clip(0.35 + 25000 / max(pts_r.shape[0], 1) * 0.20, 0.25, 0.8))

    for i in range(len(slices) - 1):
        sl = pts_r[(y >= slices[i]) & (y < slices[i + 1])]
        if sl.shape[0] < 30:
            continue

        # "supprimer épaules" mais sans détruire la symétrie :
        # on garde le dos (z haut) avec un seuil un peu plus haut, et fallback si trop peu.
        if remove_shoulders:
            thr = np.percentile(sl[:, 2], 80)
            m = sl[:, 2] >= thr
            if np.count_nonzero(m) >= 40:
                sl_use = sl[m]
            else:
                sl_use = sl
        else:
            sl_use = sl

        # profil en bins (anti densité)
        xc, zc = slice_profile_surface_binned(sl_use, cell_cm=cell_cm, z_percentile=95)

        if xc is None:
            # fallback density-free : bins sur X sans z(x)
            # on fait une discrétisation X et on prend la médiane des bins non vides
            x = sl_use[:, 0]
            xmin, xmax = np.percentile(x, [2, 98])
            if xmax - xmin < 1e-6:
                continue
            nb = max(25, int(np.ceil((xmax - xmin) / cell_cm)))
            edges = np.linspace(xmin, xmax, nb + 1)
            centers = []
            for b in range(nb):
                m = (x >= edges[b]) & (x < edges[b + 1])
                if np.count_nonzero(m) >= 3:
                    centers.append(0.5 * (edges[b] + edges[b + 1]))
            if len(centers) < 8:
                continue
            x0 = float(np.median(np.array(centers)))
        else:
            # léger lissage du profil (sur bins, donc ok)
            if len(zc) >= 11:
                zc_s = savgol_filter(zc, 11, 2)
            else:
                zc_s = zc

            # centre par symétrie (le plus “médian” géométriquement)
            x0 = symmetry_center_1d(xc, zc_s)

            # si la symétrie donne un truc bizarre, fallback médiane bins
            if not np.isfinite(x0):
                x0 = robust_center_from_bins(xc)
            if x0 is None:
                continue

        y_mid = float(np.median(sl_use[:, 1]))  # médiane (pas moyenne)
        z0 = float(np.percentile(sl_use[:, 2], 90))

        # continuité : ne pas verrouiller trop tôt
        if prev_x is not None and len(spine) > 12:
            if abs(x0 - prev_x) > 2.2:
                x0 = prev_x
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

    remove_shoulders = st.toggle("Supprimer épaules (haut)", True)

    st.subheader("🧽 Lissage")
    do_smooth = st.toggle("Activer", True)
    strong_smooth = st.toggle("Lissage fort (anti-pics)", True)
    smooth_window = st.slider("Fenêtre lissage", 5, 151, 91, step=2)
    median_k = st.slider("Anti-pics (médian)", 3, 31, 11, step=2)

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm

        # nettoyage Y
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 5)) & (pts[:, 1] < np.percentile(pts[:, 1], 95))
        pts = pts[mask]

        # centrage global X (affichage uniquement, OK)
        pts[:, 0] -= np.median(pts[:, 0])

        # extraction corrigée anti-biais densité
        spine = extract_spine_density_free(pts, remove_shoulders=remove_shoulders)

        # si trop peu, relâche épaules
        if spine.shape[0] < 10 and remove_shoulders:
            spine = extract_spine_density_free(pts, remove_shoulders=False)

        if spine.shape[0] < 8:
            # dernier recours : density-free par bins X (pas moyenne)
            y = pts[:, 1]
            slices = np.linspace(np.percentile(y, 10), np.percentile(y, 92), 90)
            tmp_sp = []
            for i in range(len(slices) - 1):
                sl = pts[(y >= slices[i]) & (y < slices[i + 1])]
                if sl.shape[0] < 20:
                    continue
                x = sl[:, 0]
                xmin, xmax = np.percentile(x, [2, 98])
                if xmax - xmin < 1e-6:
                    continue
                cell = 0.6
                nb = max(20, int(np.ceil((xmax - xmin) / cell)))
                edges = np.linspace(xmin, xmax, nb + 1)
                centers = []
                for b in range(nb):
                    m = (x >= edges[b]) & (x < edges[b + 1])
                    if np.count_nonzero(m) >= 3:
                        centers.append(0.5 * (edges[b] + edges[b + 1]))
                if len(centers) < 8:
                    continue
                x0 = float(np.median(np.array(centers)))
                y0 = float(np.median(sl[:, 1]))
                z0 = float(np.percentile(sl[:, 2], 90))
                tmp_sp.append([x0, y0, z0])
            spine = np.array(tmp_sp, dtype=float) if len(tmp_sp) else np.empty((0, 3), dtype=float)

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
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.4)
        ax_f.set_title("Frontale", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.4)
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
                Extraction anti-biais densité: profil en bins X (1 bin = 1 vote) + centre par symétrie.
                Aucune moyenne des points bruts.
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
