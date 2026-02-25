import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
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
.badge-ok {display:inline-block; padding:2px 8px; border-radius:999px; background:#e8f7ee; color:#156f3b; font-weight:700; font-size:0.85rem;}
.badge-no {display:inline-block; padding:2px 8px; border-radius:999px; background:#fdecec; color:#9b1c1c; font-weight:700; font-size:0.85rem;}
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
        ["Flèche Lombaire", f"{results['fl']:.2f} cm ({results['fl_status']})"],
        ["Déviation Latérale Max", f"{results['dev_f']:.2f} cm"],
        ["Angle Lordose Lombaire (est.)", f"{results['lordosis_deg']:.1f}° ({results['lordosis_status']})"],
        ["Angle Cyphose Dorsale (est.)", f"{results['kyphosis_deg']:.1f}° ({results['kyphosis_status']})"],
        ["Jonction Thoraco-Lombaire (est.)", f"{results['y_junction']:.1f} cm" if results['y_junction'] is not None else "n/a"],
        ["Couverture / Fiabilité", f"{results['coverage_pct']:.0f}% / {results['reliability_pct']:.0f}%"],
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
# METRICS
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

def classify_fl(fl_cm, lo=2.5, hi=4.5):
    if fl_cm < lo:
        return "Trop faible"
    if fl_cm > hi:
        return "Trop élevée"
    return "Normale"

def classify_angle(val_deg, lo, hi):
    if val_deg < lo:
        return "Trop faible"
    if val_deg > hi:
        return "Trop élevée"
    return "Normale"

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
# MIDLINE + QUALITY (anti-biais densité)
# ==============================
def slice_profile_surface_binned(sl, cell_cm=0.70, z_percentile=92, min_pts_per_bin=2):
    x = sl[:, 0]
    z = sl[:, 2]
    xmin, xmax = np.percentile(x, [2, 98])
    if xmax - xmin < 1e-6:
        return None, None, 0.0

    nbins = max(26, int(np.ceil((xmax - xmin) / cell_cm)))
    edges = np.linspace(xmin, xmax, nbins + 1)

    xc, zc = [], []
    for b in range(nbins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if np.count_nonzero(m) < min_pts_per_bin:
            continue
        xc.append(0.5 * (edges[b] + edges[b + 1]))
        zc.append(float(np.percentile(z[m], z_percentile)))

    if len(xc) < 10:
        return None, None, 0.0

    xc = np.array(xc, dtype=float)
    zc = np.array(zc, dtype=float)
    o = np.argsort(xc)
    xc, zc = xc[o], zc[o]

    width = float(np.percentile(xc, 95) - np.percentile(xc, 5))
    return xc, zc, width

def midline_from_profile(xc, zc, q_edge=8):
    xl = float(np.percentile(xc, q_edge))
    xr = float(np.percentile(xc, 100 - q_edge))
    x_mid = 0.5 * (xl + xr)
    z_mid = float(np.interp(x_mid, xc, zc))
    return x_mid, z_mid

def quality_score(n_points_slice, n_bins, width_cm, jump_cm, expected_width=(10.0, 45.0)):
    s_pts = np.clip((n_points_slice - 30) / 200.0, 0.0, 1.0)
    s_bins = np.clip((n_bins - 10) / 25.0, 0.0, 1.0)

    w_lo, w_hi = expected_width
    if width_cm <= 0:
        s_w = 0.0
    elif width_cm < w_lo:
        s_w = np.clip(width_cm / w_lo, 0.0, 1.0)
    elif width_cm > w_hi:
        s_w = np.clip(w_hi / width_cm, 0.0, 1.0)
    else:
        s_w = 1.0

    if jump_cm is None:
        s_j = 1.0
    else:
        s_j = np.clip(1.0 - (abs(jump_cm) / 4.0), 0.0, 1.0)

    score = 0.25 * s_pts + 0.35 * s_bins + 0.25 * s_w + 0.15 * s_j
    return float(np.clip(score, 0.0, 1.0))

def extract_midline_full_with_quality(
    pts,
    remove_lateral_outliers=True,
    cell_cm=0.70,
    z_percentile=92,
    q_edge=8,
    y_low=2,
    y_high=98,
    allow_fill_gaps=True
):
    R = estimate_rotation_xz(pts)
    pr = apply_rotation_xz(pts, R)

    y = pr[:, 1]
    y0 = np.percentile(y, y_low)
    y1 = np.percentile(y, y_high)

    n_slices = int(np.clip((pr.shape[0] // 2500) + 240, 220, 360))
    edges_y = np.linspace(y0, y1, n_slices)

    spine = []
    qlist = []
    prev_x = None
    prev_z = None

    for i in range(len(edges_y) - 1):
        sl = pr[(y >= edges_y[i]) & (y < edges_y[i + 1])]
        y_mid = float(0.5 * (edges_y[i] + edges_y[i + 1]))

        if sl.shape[0] < 25:
            if allow_fill_gaps and prev_x is not None and prev_z is not None:
                spine.append([prev_x, y_mid, prev_z])
                qlist.append(0.15)
            continue

        if remove_lateral_outliers:
            x = sl[:, 0]
            xl, xr = np.percentile(x, [1.5, 98.5])
            sl = sl[(x >= xl) & (x <= xr)]
            if sl.shape[0] < 20:
                if allow_fill_gaps and prev_x is not None and prev_z is not None:
                    spine.append([prev_x, y_mid, prev_z])
                    qlist.append(0.15)
                continue

        xc, zc, width = slice_profile_surface_binned(sl, cell_cm=cell_cm, z_percentile=z_percentile, min_pts_per_bin=2)
        if xc is None:
            xc, zc, width = slice_profile_surface_binned(sl, cell_cm=min(1.10, cell_cm * 1.5), z_percentile=z_percentile, min_pts_per_bin=2)

        if xc is None:
            if allow_fill_gaps and prev_x is not None and prev_z is not None:
                spine.append([prev_x, y_mid, prev_z])
                qlist.append(0.15)
            continue

        if len(zc) >= 9:
            zc_s = savgol_filter(zc, 9, 2)
        else:
            zc_s = zc

        x0, z0 = midline_from_profile(xc, zc_s, q_edge=q_edge)

        jump = None if prev_x is None else (x0 - prev_x)
        if prev_x is not None and abs(jump) > 3.0:
            x0 = prev_x
            z0 = float(np.interp(x0, xc, zc_s))
            jump = 0.0

        score = quality_score(
            n_points_slice=int(sl.shape[0]),
            n_bins=int(len(xc)),
            width_cm=float(width),
            jump_cm=jump
        )

        prev_x, prev_z = x0, z0
        spine.append([x0, float(np.median(sl[:, 1])), float(z0)])
        qlist.append(score)

    if len(spine) == 0:
        return np.empty((0, 3), dtype=float), np.array([], dtype=float)

    spine = np.array(spine, dtype=float)
    qlist = np.array(qlist, dtype=float)

    o = np.argsort(spine[:, 1])
    spine = spine[o]
    qlist = qlist[o]

    XZ_back = spine[:, [0, 2]] @ R
    spine[:, 0] = XZ_back[:, 0]
    spine[:, 2] = XZ_back[:, 1]
    return spine, qlist

# ==============================
# ANGLES V2 (concavité/convexité + tangentes)
# ==============================
def estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=21):
    if spine.shape[0] < 25:
        return 0.0, 0.0, None

    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)

    n = len(z)
    w = int(smooth_win)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    w = max(7, w)

    z_s = savgol_filter(z, w, 3)

    dz = np.gradient(z_s, y)
    d2z = np.gradient(dz, y)

    y20 = np.percentile(y, 20)
    y80 = np.percentile(y, 80)
    low = d2z[y <= y20]
    high = d2z[y >= y80]
    if low.size < 3 or high.size < 3:
        return 0.0, 0.0, None

    sign_low = np.sign(np.median(low))
    sign_high = np.sign(np.median(high))

    # On veut bas concave (+) et haut convexe (-). Si inversé, on inverse.
    if (sign_low < 0 and sign_high > 0):
        d2z = -d2z
        dz = -dz

    y_mid_lo = np.percentile(y, 35)
    y_mid_hi = np.percentile(y, 65)
    mid_mask = (y >= y_mid_lo) & (y <= y_mid_hi)
    idx_mid = np.where(mid_mask)[0]
    if idx_mid.size == 0:
        return 0.0, 0.0, None

    j = idx_mid[np.argmin(np.abs(d2z[idx_mid]))]
    y_j = float(y[j])

    y_bot = float(np.percentile(y, 8))
    y_top = float(np.percentile(y, 92))

    i_bot = int(np.argmin(np.abs(y - y_bot)))
    i_top = int(np.argmin(np.abs(y - y_top)))
    i_j = int(j)

    theta = np.degrees(np.arctan(dz))

    lordosis = float(abs(theta[i_j] - theta[i_bot]))
    kyphosis = float(abs(theta[i_top] - theta[i_j]))

    if (y_j - y_bot) < 0.15 * (y_top - y_bot) or (y_top - y_j) < 0.15 * (y_top - y_bot):
        y_j_fb = float(np.percentile(y, 50))
        i_j_fb = int(np.argmin(np.abs(y - y_j_fb)))
        lordosis = float(abs(theta[i_j_fb] - theta[i_bot]))
        kyphosis = float(abs(theta[i_top] - theta[i_j_fb]))
        y_j = y_j_fb

    return lordosis, kyphosis, y_j

# ==============================
# COLORED CURVE
# ==============================
def plot_colored_curve(ax, x, y, q, lw=2.6):
    if len(x) < 2:
        return
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    q_seg = 0.5 * (q[:-1] + q[1:])
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(q_seg)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.autoscale_view()

# ==============================
# UI
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.divider()

    st.subheader("🧭 Midline (tout le dos) + fiabilité")
    remove_lat = st.toggle("Limiter artefacts latéraux (bras)", True)
    allow_fill = st.toggle("Remplir les manques (faible fiabilité)", True)

    cell_cm = st.slider("Taille bin X (cm)", 0.30, 1.20, 0.70, step=0.05)
    z_perc = st.slider("Surface dos (percentile Z)", 85, 99, 92, step=1)
    q_edge = st.slider("Bords (quantile X)", 5, 20, 8, step=1)

    st.divider()
    st.subheader("🧽 Lissage final (courbe)")
    do_smooth = st.toggle("Activer", True)
    strong_smooth = st.toggle("Lissage fort (anti-pics)", True)
    smooth_window = st.slider("Fenêtre lissage", 5, 151, 91, step=2)
    median_k = st.slider("Anti-pics (médian)", 3, 31, 11, step=2)

    st.divider()
    st.subheader("📐 Angles (plan sagittal) + normes")
    angle_smooth = st.slider("Lissage angles (fenêtre)", 7, 41, 21, step=2)

    show_norms = st.toggle("Afficher normes", True)

    st.caption("Normes (par défaut)")
    lord_lo, lord_hi = 40.0, 60.0
    kyph_lo, kyph_hi = 27.0, 47.0
    st.write(f"- Lordose: {lord_lo:.0f}° à {lord_hi:.0f}°")
    st.write(f"- Cyphose: {kyph_lo:.0f}° à {kyph_hi:.0f}°")

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro — Midline + Fiabilité + Angles (V2)")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm

        mask = (pts[:, 1] > np.percentile(pts[:, 1], 1)) & (pts[:, 1] < np.percentile(pts[:, 1], 99))
        pts = pts[mask]

        pts[:, 0] -= np.median(pts[:, 0])  # centrage affichage

        spine, q = extract_midline_full_with_quality(
            pts,
            remove_lateral_outliers=remove_lat,
            cell_cm=float(cell_cm),
            z_percentile=float(z_perc),
            q_edge=int(q_edge),
            y_low=2,
            y_high=98,
            allow_fill_gaps=allow_fill,
        )

        if spine.shape[0] == 0:
            st.error("Impossible d'extraire une courbe (scan trop incomplet).")
            st.stop()

        if do_smooth:
            spine = smooth_spine(spine, window=smooth_window, strong=strong_smooth, median_k=median_k)

        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        fl_status = classify_fl(fl, 2.5, 4.5)
        dev_f = float(np.max(np.abs(spine[:, 0]))) if spine.size else 0.0

        lordosis_deg, kyphosis_deg, y_junction = estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=int(angle_smooth))

        lordosis_status = classify_angle(lordosis_deg, lord_lo, lord_hi)
        kyphosis_status = classify_angle(kyphosis_deg, kyph_lo, kyph_hi)

        y_span_pts = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
        y_span_sp = float(np.max(spine[:, 1]) - np.min(spine[:, 1])) if spine.shape[0] else 0.0
        coverage_pct = 100.0 * (y_span_sp / y_span_pts) if y_span_pts > 1e-6 else 0.0
        reliability_pct = 100.0 * float(np.mean(q >= 0.6)) if q.size else 0.0

        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.4, 4.2))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        plot_colored_curve(ax_f, spine[:, 0], spine[:, 1], q, lw=2.8)
        ax_f.set_title("Frontale (couleur = fiabilité)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=170)

        fig_s, ax_s = plt.subplots(figsize=(2.4, 4.2))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        plot_colored_curve(ax_s, spine[:, 2], spine[:, 1], q, lw=2.8)
        if vertical_z.size:
            ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        if y_junction is not None:
            ax_s.axhline(y_junction, linestyle="--", linewidth=1, alpha=0.6)
        ax_s.set_title("Sagittale (couleur = fiabilité)", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=170)

        st.write("### 📈 Analyse Visuelle")
        _, c1, c2, _ = st.columns([1, 1, 1, 1])
        c1.pyplot(fig_f)
        c2.pyplot(fig_s)

        st.write("### 🎨 Légende fiabilité")
        st.caption("Rouge = faible | Jaune = moyen | Vert = fiable (score 0→1)")
        fig_leg, ax_leg = plt.subplots(figsize=(5.0, 0.35))
        ax_leg.set_axis_off()
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax_leg.imshow(gradient, aspect="auto", cmap="RdYlGn")
        st.pyplot(fig_leg)

        badge_fl = '<span class="badge-ok">Normale</span>' if fl_status == "Normale" else '<span class="badge-no">Hors norme</span>'
        badge_lord = '<span class="badge-ok">Normale</span>' if lordosis_status == "Normale" else '<span class="badge-no">Hors norme</span>'
        badge_kyph = '<span class="badge-ok">Normale</span>' if kyphosis_status == "Normale" else '<span class="badge-no">Hors norme</span>'

        st.write("### 📋 Synthèse des résultats")
        st.markdown(f"""
        <div class="result-box">
            <p><b>📏 Flèche Dorsale :</b> <span class="value-text">{fd:.2f} cm</span></p>
            <p><b>📏 Flèche Lombaire :</b> <span class="value-text">{fl:.2f} cm</span> &nbsp; {badge_fl}
               <br><span style="color:#666;font-size:0.9rem;">Référence: 2.5 à 4.5 cm</span></p>
            <p><b>↔️ Déviation Latérale Max :</b> <span class="value-text">{dev_f:.2f} cm</span></p>
            <p><b>📐 Angle lordose lombaire (est.) :</b> <span class="value-text">{lordosis_deg:.1f}°</span>
               {"&nbsp; " + badge_lord if show_norms else ""}
               {"<br><span style='color:#666;font-size:0.9rem;'>Référence: 40° à 60°</span>" if show_norms else ""}</p>
            <p><b>📐 Angle cyphose dorsale (est.) :</b> <span class="value-text">{kyphosis_deg:.1f}°</span>
               {"&nbsp; " + badge_kyph if show_norms else ""}
               {"<br><span style='color:#666;font-size:0.9rem;'>Référence: 27° à 47°</span>" if show_norms else ""}</p>
            <p><b>🔁 Jonction thoraco-lombaire (est.) :</b> <span class="value-text">{(f"{y_junction:.1f} cm" if y_junction is not None else "n/a")}</span></p>
            <p><b>✅ Couverture hauteur :</b> <span class="value-text">{coverage_pct:.0f}%</span>
               &nbsp; <b>Fiabilité :</b> <span class="value-text">{reliability_pct:.0f}%</span>
               <br><span style="color:#666;font-size:0.9rem;">Fiabilité = % des points avec score ≥ 0.60</span></p>
            <div class="disclaimer">
                Midline “tout le dos” = milieu entre bords (bins X, anti-biais densité) + couleur selon score qualité.
                Angles V2 = plan sagittal via concavité/convexité (z''(y)) + différence de tangentes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        res = {
            "fd": float(fd),
            "fl": float(fl),
            "fl_status": fl_status,
            "dev_f": float(dev_f),
            "lordosis_deg": float(lordosis_deg),
            "kyphosis_deg": float(kyphosis_deg),
            "lordosis_status": lordosis_status,
            "kyphosis_status": kyphosis_status,
            "y_junction": None if y_junction is None else float(y_junction),
            "coverage_pct": float(coverage_pct),
            "reliability_pct": float(reliability_pct),
        }
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
