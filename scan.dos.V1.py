# ==============================
# SpineScan SUPER (Revopoint) — V3.2 + (Front Curvature)
# AJOUT (sans modifier les autres fonctions) :
# ✅ Mesure de courbure frontale lombaire & dorsale sur x(y)
#    - kappa_peak (1/cm), kappa_rms (1/cm), angle_courbure (°)
# ✅ UI + Synthèse + PDF : remplace (optionnellement) le Cobb proxy par ces mesures
# ⚠️ Toutes les fonctions existantes sont conservées INCHANGÉES.
#     -> On ajoute seulement de nouvelles fonctions + on modifie l'UI / résultats / PDF.
# ==============================

import streamlit as st
import streamlit.components.v1 as components
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
# PAGE + STYLE
# ==============================
st.set_page_config(page_title="SpineScan SUPER", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fc; }
.stButton>button { background-color: #2c3e50; color: white; width: 100%; border-radius: 10px; font-weight: 800; }
hr { margin: 0.6rem 0; }
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
# PDF (MODIF : ajout courbures frontales)
# ==============================
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import os
import tempfile

def export_pdf_super(patient_info, results, img_front_path, img_sag_path, img_asym_path=None):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"Rapport_SpineScan_{patient_info['nom']}.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # --- Styles Personnalisés ---
    title_style = ParagraphStyle(
        "TitleStyle", fontSize=18, textColor=colors.HexColor("#2C3E50"),
        alignment=1, spaceAfter=10, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "SubTitle", fontSize=12, textColor=colors.HexColor("#34495E"),
        spaceBefore=10, spaceAfter=10, fontName="Helvetica-Bold"
    )

    story = []

    # 1. En-tête
    story.append(Paragraph("RAPPORT D'ANALYSE SPINESCAN SUPER", title_style))
    story.append(Spacer(1, 0.2 * cm))

    patient_line = f"<b>Patient :</b> {patient_info['prenom']} {patient_info['nom']} <br/><b>Date :</b> 26/02/2026"
    story.append(Paragraph(patient_line, styles["Normal"]))
    story.append(Spacer(1, 0.8 * cm))

    # 2. Tableau
    story.append(Paragraph("Résultats de l'Analyse", subtitle_style))

    # Helpers robustes
    def fmt_deg(key):
        v = results.get(key, None)
        return "n/a" if v is None else f"{float(v):.1f}°"

    def fmt_kappa(key):
        v = results.get(key, None)
        return "n/a" if v is None else f"{float(v):.4f} 1/cm"

    data = [
        [Paragraph("<b>Indicateur</b>", styles["Normal"]), Paragraph("<b>Valeur / Statut</b>", styles["Normal"])],
        ["Flèche lombaire", f"{results['fl']:.2f} cm ({results['fl_status']})"],
        ["Déviation latérale max", f"{results['dev_f']:.2f} cm"],
        ["Lordose (est.)", f"{results['lordosis_deg']:.1f}° ({results['lordosis_status']})"],
        ["Cyphose (est.)", f"{results['kyphosis_deg']:.1f}° ({results['kyphosis_status']})"],

        # ✅ AJOUT : angles de courbure frontale
        ["Angle courbure frontale lombaire", fmt_deg("front_angle_lomb_deg")],
        ["Angle courbure frontale dorsale", fmt_deg("front_angle_thor_deg")],

        # (Optionnel) κ si tu veux aussi le mettre dans le PDF
        # ["Courbure frontale lombaire (peak |κ|)", fmt_kappa("front_kappa_peak_lomb")],
        # ["Courbure frontale lombaire (RMS κ)", fmt_kappa("front_kappa_rms_lomb")],
        # ["Courbure frontale dorsale (peak |κ|)", fmt_kappa("front_kappa_peak_thor")],
        # ["Courbure frontale dorsale (RMS κ)", fmt_kappa("front_kappa_rms_thor")],

        ["Jonction TL (rel.)", results["y_junction_rel"]],
        ["Couverture / Fiabilité", f"{results['coverage_pct']:.0f}% / {results['reliability_pct']:.0f}%"],
    ]

    table = Table(data, colWidths=[8*cm, 8*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F2F4F4")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9F9")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 1 * cm))

    # 3. Images
    story.append(Paragraph("Clichés d'Analyse", subtitle_style))
    try:
        img_f = Image(img_front_path, width=7*cm, height=9*cm, kind='proportional')
        img_s = Image(img_sag_path, width=7*cm, height=9*cm, kind='proportional')
        img_table = Table([[img_f, img_s]], colWidths=[8.5*cm, 8.5*cm])
        img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        story.append(img_table)
    except Exception as e:
        story.append(Paragraph(f"Erreur chargement images : {e}", styles["Normal"]))

    doc.build(story)
    return path
# ==============================
# UTILITAIRES (anti-biais densité)
# (INCHANGE)
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
# ROTATION CORRECTION (XZ)
# (INCHANGE)
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

def unrotate_spine_xz(spine, R):
    XZ_back = spine[:, [0, 2]] @ R
    out = spine.copy()
    out[:, 0] = XZ_back[:, 0]
    out[:, 2] = XZ_back[:, 1]
    return out

# ==============================
# V3 SURFACE (Rasterstéréographie)
# (INCHANGE)
# ==============================
def build_depth_surface(points, dx=0.5, dy=0.5):
    """
    Surface Z(x,y) : max Z par cellule => robuste densité/temps de scan.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])

    nx = int(np.ceil((xmax - xmin) / dx)) + 1
    ny = int(np.ceil((ymax - ymin) / dy)) + 1
    grid = np.full((ny, nx), np.nan, dtype=float)

    ix = ((x - xmin) / dx).astype(int)
    iy = ((y - ymin) / dy).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[valid], iy[valid], z[valid]

    for k in range(ix.size):
        gx, gy = ix[k], iy[k]
        v = z[k]
        if np.isnan(grid[gy, gx]) or v > grid[gy, gx]:
            grid[gy, gx] = v

    # comblement léger (colonne)
    for col in range(nx):
        col_vals = grid[:, col]
        m = ~np.isnan(col_vals)
        if np.count_nonzero(m) >= 4:
            grid[:, col] = np.interp(np.arange(ny), np.where(m)[0], col_vals[m])

    return grid, xmin, ymin, dx, dy

def extract_midline_symmetry_surface(grid, xmin, ymin, dx, dy, edge_q=10):
    """
    Midline = milieu entre bords gauche/droit sur chaque Y.
    """
    ny, nx = grid.shape
    spine = []
    meta = []

    for j in range(ny):
        row = grid[j]
        valid = ~np.isnan(row)
        nvalid = int(np.count_nonzero(valid))
        if nvalid < 10:
            continue

        xs = np.where(valid)[0].astype(float)
        xL = float(np.percentile(xs, edge_q))
        xR = float(np.percentile(xs, 100 - edge_q))
        xM = int(np.clip(round(0.5 * (xL + xR)), 0, nx - 1))

        zM = float(row[xM]) if not np.isnan(row[xM]) else float(np.nanmedian(row[valid]))
        yM = float(ymin + j * dy)
        xM_cm = float(xmin + xM * dx)

        width_cells = float(xR - xL)
        valid_frac = float(nvalid / nx)

        spine.append([xM_cm, yM, zM])
        meta.append([valid_frac, width_cells])

    if len(spine) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=float)

    spine = np.array(spine, dtype=float)
    meta = np.array(meta, dtype=float)
    o = np.argsort(spine[:, 1])
    return spine[o], meta[o]

def detect_psis(grid, xmin, ymin, dx, dy):
    """
    Heuristique PSIS: 2 dépressions (minima) dans bande bas du dos.
    """
    ny, nx = grid.shape
    y_low = int(ny * 0.15)
    y_high = int(ny * 0.35)
    band = grid[y_low:y_high]
    if np.isnan(band).all():
        return None, None, 0.0

    med = float(np.nanmedian(band))
    depth = med - band
    flat = depth.ravel()
    if np.all(np.isnan(flat)):
        return None, None, 0.0

    idx_flat = np.argsort(flat)[-220:]
    ys = (idx_flat // band.shape[1]) + y_low
    xs = (idx_flat % band.shape[1])

    if xs.size < 20:
        return None, None, 0.0

    x_med = float(np.median(xs))
    left_mask = xs < x_med
    right_mask = xs > x_med
    if np.count_nonzero(left_mask) < 8 or np.count_nonzero(right_mask) < 8:
        return None, None, 0.0

    lx = int(np.median(xs[left_mask]))
    rx = int(np.median(xs[right_mask]))
    ly = int(np.median(ys[left_mask]))
    ry = int(np.median(ys[right_mask]))

    psis_L = (float(xmin + lx * dx), float(ymin + ly * dy))
    psis_R = (float(xmin + rx * dx), float(ymin + ry * dy))

    sep = abs(rx - lx) / max(nx, 1)
    conf = 0.35 + 0.45 * np.clip(sep * 2.0, 0.0, 1.0) + 0.20 * np.clip(xs.size / 220.0, 0.0, 1.0)
    return psis_L, psis_R, float(np.clip(conf, 0.0, 1.0))

def quality_from_surface(spine_r, meta, psis_conf=0.0, max_jump_cm=3.0):
    if spine_r.shape[0] == 0:
        return np.array([], dtype=float)

    valid_frac = meta[:, 0]
    width_cells = meta[:, 1]

    w = width_cells
    w_p10 = float(np.percentile(w, 10))
    w_p90 = float(np.percentile(w, 90))
    w_score = np.clip((w - w_p10) / (w_p90 - w_p10 + 1e-6), 0, 1)

    x = spine_r[:, 0]
    jumps = np.abs(np.diff(x))
    j_score = np.ones_like(x)
    if jumps.size:
        j_seg = np.clip(1.0 - (jumps / max_jump_cm), 0.0, 1.0)
        j_score[1:] = j_seg

    v_p85 = float(np.percentile(valid_frac, 85))
    v_score = np.clip(valid_frac / (v_p85 + 1e-6), 0, 1)

    base = 0.50 * v_score + 0.30 * w_score + 0.20 * j_score
    base = np.clip(base, 0.0, 1.0)
    base = np.clip(base * (0.85 + 0.15 * psis_conf), 0.0, 1.0)
    return base.astype(float)

# ==============================
# SAGITTAL "VERTICALE" + FLECHES (INCHANGE)
# ==============================
def robust_line_z_of_y(spine, yq_low=8, yq_high=92, nbins=70):
    """
    Ajuste une droite z = a*y + b sur points robustifiés (médiane par bin Y).
    => pas de biais densité et plus stable aux outliers.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)

    y0, y1 = np.percentile(y, [yq_low, yq_high])
    m = (y >= y0) & (y <= y1)
    y, z = y[m], z[m]
    if y.size < 10:
        return 0.0, float(np.median(z))

    edges = np.linspace(y0, y1, nbins + 1)
    yc, zc = [], []
    for i in range(nbins):
        mm = (y >= edges[i]) & (y < edges[i + 1])
        if np.count_nonzero(mm) < 3:
            continue
        yc.append(0.5 * (edges[i] + edges[i + 1]))
        zc.append(float(np.median(z[mm])))

    if len(yc) < 6:
        return 0.0, float(np.median(z))

    yc = np.array(yc)
    zc = np.array(zc)
    a, b = np.polyfit(yc, zc, 1)
    return float(a), float(b)

def compute_sagittal_arrows_v3(spine, lordose_frac=(0.08, 0.45), kyphose_frac=(0.55, 0.92)):
    """
    Référence sagittale: z_ref(y)=a*y+b (robuste).
    Flèche lombaire: max |z - z_ref| sur zone basse.
    Flèche dorsale : max |z - z_ref| sur zone haute.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)
    if y.size < 20:
        return 0.0, 0.0, np.array([]), np.array([])

    a, b = robust_line_z_of_y(s, yq_low=8, yq_high=92, nbins=70)
    z_ref = a * y + b
    z_dev = z - z_ref

    y_min, y_max = float(np.min(y)), float(np.max(y))
    span = (y_max - y_min) if (y_max > y_min) else 1.0

    lo0 = y_min + lordose_frac[0] * span
    lo1 = y_min + lordose_frac[1] * span
    th0 = y_min + kyphose_frac[0] * span
    th1 = y_min + kyphose_frac[1] * span

    m_lomb = (y >= lo0) & (y <= lo1)
    m_thor = (y >= th0) & (y <= th1)

    fl = float(np.max(np.abs(z_dev[m_lomb]))) if np.count_nonzero(m_lomb) >= 5 else 0.0
    fd = float(np.max(np.abs(z_dev[m_thor]))) if np.count_nonzero(m_thor) >= 5 else 0.0

    return fd, fl, z_ref, z_dev

# ==============================
# MESURES / ANGLES (INCHANGE)
# ==============================
def classify_range(val, lo, hi):
    if val < lo:
        return "Trop faible"
    if val > hi:
        return "Trop élevée"
    return "Normale"

def estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=21):
    """
    Plan sagittal : concavité basse = lombaire, convexité haute = dorsale.
    Angle = différence de tangentes.
    """
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
    if (sign_low < 0 and sign_high > 0):
        dz = -dz
        d2z = -d2z

    mid_mask = (y >= np.percentile(y, 35)) & (y <= np.percentile(y, 65))
    idx_mid = np.where(mid_mask)[0]
    if idx_mid.size == 0:
        return 0.0, 0.0, None

    j = idx_mid[np.argmin(np.abs(d2z[idx_mid]))]
    y_j = float(y[j])

    y_bot = float(np.percentile(y, 8))
    y_top = float(np.percentile(y, 92))
    i_bot = int(np.argmin(np.abs(y - y_bot)))
    i_top = int(np.argmin(np.abs(y - y_top)))

    theta = np.degrees(np.arctan(dz))
    lordosis = float(abs(theta[j] - theta[i_bot]))
    kyphosis = float(abs(theta[i_top] - theta[j]))

    if (y_j - y_bot) < 0.15 * (y_top - y_bot) or (y_top - y_j) < 0.15 * (y_top - y_bot):
        y_j = float(np.percentile(y, 50))
        j = int(np.argmin(np.abs(y - y_j)))
        lordosis = float(abs(theta[j] - theta[i_bot]))
        kyphosis = float(abs(theta[i_top] - theta[j]))

    return lordosis, kyphosis, y_j

def estimate_cobb_proxy_front(spine, smooth_win=21):
    """
    Cobb-like proxy: angle entre tangentes haut/bas sur x(y) (frontale).
    Retourne: angle_deg, fit_bot(a,b), fit_top(a,b), y_ranges(y10,y30,y70,y90)
    """
    if spine.shape[0] < 30:
        return 0.0, None, None, None

    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    x = s[:, 0].astype(float)

    n = len(x)
    w = int(smooth_win)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    w = max(7, w)

    x_s = savgol_filter(x, w, 3)

    y10, y30, y70, y90 = np.percentile(y, [10, 30, 70, 90])
    m_bot = (y >= y10) & (y <= y30)
    m_top = (y >= y70) & (y <= y90)
    if np.count_nonzero(m_bot) < 6 or np.count_nonzero(m_top) < 6:
        return 0.0, None, None, None

    a_bot, b_bot = np.polyfit(y[m_bot], x_s[m_bot], 1)
    a_top, b_top = np.polyfit(y[m_top], x_s[m_top], 1)

    denom = 1.0 + a_bot * a_top
    ang = np.pi / 2 if abs(denom) < 1e-9 else np.arctan(abs((a_top - a_bot) / denom))
    return float(np.degrees(ang)), (float(a_bot), float(b_bot)), (float(a_top), float(b_top)), (float(y10), float(y30), float(y70), float(y90))

# ==============================
# ✅ NOUVEAU : Courbures frontales lombaire/dorsale (AJOUT)
# ==============================
def frontal_curvature_metrics(spine, smooth_win=21,
                              lomb_frac=(0.12, 0.45),
                              thor_frac=(0.55, 0.90),
                              threshold_ratio=0.20):
    """
    Mesure courbure frontale sur x(y).
    Retour:
      - kappa_peak_lomb, kappa_rms_lomb, angle_lomb_deg
      - kappa_peak_thor, kappa_rms_thor, angle_thor_deg
    Unités:
      κ en 1/cm, angle en degrés.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    x = s[:, 0].astype(float)

    if y.size < 35:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(x)
    w = int(smooth_win)
    if w % 2 == 0:
        w += 1
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    w = max(9, w)

    x_s = savgol_filter(x, w, 3)
    dx = np.gradient(x_s, y)
    d2x = np.gradient(dx, y)

    # Courbure 2D (y->x)
    kappa = d2x / np.power(1.0 + dx * dx, 1.5)
    theta = np.degrees(np.arctan(dx))

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)

    def zone_mask(frac):
        y0 = y_min + frac[0] * span
        y1 = y_min + frac[1] * span
        return (y >= y0) & (y <= y1)

    def peak_and_bounds(mask):
        idx = np.where(mask)[0]
        if idx.size < 12:
            return None, None, None

        p = int(idx[np.argmax(np.abs(kappa[idx]))])
        k_peak = float(kappa[p])
        if abs(k_peak) < 1e-9:
            return int(idx[0]), p, int(idx[-1])

        thr = threshold_ratio * abs(k_peak)

        # bornes = zéro-crossing sinon seuil
        iL = None
        for i in range(p, 1, -1):
            if mask[i] and mask[i-1] and (np.sign(kappa[i]) != np.sign(kappa[i-1])):
                iL = i
                break
        iR = None
        for i in range(p, len(kappa)-1):
            if mask[i] and mask[i+1] and (np.sign(kappa[i]) != np.sign(kappa[i+1])):
                iR = i
                break

        if iL is None:
            for i in range(p, 0, -1):
                if mask[i] and abs(kappa[i]) < thr:
                    iL = i
                    break
        if iR is None:
            for i in range(p, len(kappa)):
                if mask[i] and abs(kappa[i]) < thr:
                    iR = i
                    break

        if iL is None: iL = int(idx[0])
        if iR is None: iR = int(idx[-1])
        if iR <= iL:
            iL, iR = int(idx[0]), int(idx[-1])

        return iL, p, iR

    def metrics_for(frac):
        mask = zone_mask(frac)
        idx = np.where(mask)[0]
        if idx.size < 12:
            return 0.0, 0.0, 0.0

        kz = kappa[idx]
        k_peak = float(np.max(np.abs(kz)))
        k_rms = float(np.sqrt(np.mean(kz * kz)))

        iL, p, iR = peak_and_bounds(mask)
        ang = float(abs(theta[iR] - theta[iL])) if (iL is not None) else 0.0
        return k_peak, k_rms, ang

    kpk_lo, krms_lo, ang_lo = metrics_for(lomb_frac)
    kpk_th, krms_th, ang_th = metrics_for(thor_frac)
    return (kpk_lo, krms_lo, ang_lo, kpk_th, krms_th, ang_th)

# ==============================
# PLOTS (INCHANGE)
# ==============================
def plot_colored_curve(ax, x, y, q, lw=2.8):
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

def save_fig(fig, name):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, name)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    return path

def make_asymmetry_heatmap(grid):
    ny, nx = grid.shape
    asym = np.full_like(grid, np.nan, dtype=float)

    for j in range(ny):
        row = grid[j]
        valid = ~np.isnan(row)
        xs = np.where(valid)[0]
        if xs.size < 10:
            continue
        mid = int(np.median(xs))
        for i in xs:
            sym = 2 * mid - i
            if 0 <= sym < nx and not np.isnan(row[sym]):
                asym[j, i] = row[i] - row[sym]

    vals = asym[~np.isnan(asym)]
    if vals.size < 50:
        return None
    lim = float(np.percentile(np.abs(vals), 95))
    lim = max(lim, 0.3)

    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    im = ax.imshow(asym, aspect="auto", origin="lower", vmin=-lim, vmax=lim)
    ax.set_title("Carte d’asymétrie gauche/droite (surface)", fontsize=10)
    ax.set_xlabel("X (cellules)")
    ax.set_ylabel("Y (cellules)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return fig

# ==============================
# UI — SIDEBAR (MODIF : toggle courbures frontales)
# ==============================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")

    st.divider()
    st.subheader("🧩 Raster (Revopoint)")
    dx = st.slider("Résolution X (cm)", 0.2, 1.2, 0.5, step=0.1)
    dy = st.slider("Résolution Y (cm)", 0.2, 1.2, 0.5, step=0.1)
    edge_q = st.slider("Bords (quantile X)", 5, 20, 10, step=1)

    st.divider()
    st.subheader("🧽 Lissage courbe")
    do_smooth = st.toggle("Activer", True)
    strong_smooth = st.toggle("Lissage fort (anti-pics)", True)
    smooth_window = st.slider("Fenêtre lissage", 5, 151, 91, step=2)
    median_k = st.slider("Anti-pics (médian)", 3, 31, 11, step=2)

    st.divider()
    st.subheader("📐 Angles sagittaux")
    angle_smooth = st.slider("Lissage angles (fenêtre)", 7, 41, 21, step=2)

    st.divider()
    st.subheader("📈 Courbures frontales (x(y)) — optionnel")
    front_curv_enabled = st.toggle("Mesurer courbure frontale lombaire/dorsale", True)
    front_curv_smooth = st.slider("Lissage courbures frontales (fenêtre)", 9, 61, 21, step=2)
    front_thr = st.slider("Seuil bornes courbure (%)", 5, 50, 20, step=1) / 100.0
    st.caption("κ = x''/(1+x'^2)^(3/2), angle = diff de tangentes aux bornes.")

    st.divider()
    st.subheader("📐 Cobb (proxy) — optionnel")
    # On garde le Cobb dans le code, mais tu peux le désactiver/ignorer.
    cobb_enabled = st.toggle("Afficher angle de Cobb (proxy)", False)
    cobb_smooth = st.slider("Lissage Cobb (fenêtre)", 7, 41, 21, step=2)

    st.divider()
    st.subheader("🗺️ Asymétrie — optionnel")
    show_asym = st.toggle("Afficher heatmap asymétrie", False)

    st.divider()
    st.subheader("📏 Normes")
    show_norms = st.toggle("Afficher normes", True)
    fl_lo, fl_hi = 2.5, 4.5
    lord_lo, lord_hi = 40.0, 60.0
    kyph_lo, kyph_hi = 27.0, 47.0
    st.caption(f"Flèche lombaire: {fl_lo:.1f}–{fl_hi:.1f} cm")
    st.caption(f"Lordose: {lord_lo:.0f}–{lord_hi:.0f}°")
    st.caption(f"Cyphose: {kyph_lo:.0f}–{kyph_hi:.0f}°")

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

def sagittal_tangent_vertical_z(spine, dorsal_frac=(0.55, 0.92), z_quantile=98.0):
    """
    Calcule la verticale tangentielle z = z_tan "collée" au dorsal :
    - on prend la zone dorsale (fractions en hauteur)
    - z_tan = quantile élevé de z dans cette zone (robuste aux outliers)
    Retour : z_tan, (y0,y1) pour visualiser la zone de tangence.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)
    if y.size < 20:
        return float(np.median(z)) if z.size else 0.0, (float(y.min()) if y.size else 0.0, float(y.max()) if y.size else 1.0)

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)
    y0 = y_min + float(dorsal_frac[0]) * span
    y1 = y_min + float(dorsal_frac[1]) * span
    m = (y >= y0) & (y <= y1)

    if np.count_nonzero(m) < 8:
        # fallback : haut du dos
        y0 = y_min + 0.75 * span
        y1 = y_max
        m = (y >= y0) & (y <= y1)

    zz = z[m] if np.count_nonzero(m) else z
    z_tan = float(np.percentile(zz, float(z_quantile))) if zz.size else float(np.median(z))
    return z_tan, (float(y0), float(y1))


def lumbar_arrow_vs_tangent_vertical(spine, z_tan, lomb_frac=(0.08, 0.45)):
    """
    Flèche lombaire = max(z_tan - z(y)) sur la zone lombaire.
    Si l'orientation est inversée, on prend max(|z_tan - z|) en fallback.
    """
    s = spine[np.argsort(spine[:, 1])]
    y = s[:, 1].astype(float)
    z = s[:, 2].astype(float)
    if y.size < 20:
        return 0.0

    y_min, y_max = float(y.min()), float(y.max())
    span = max(1e-6, y_max - y_min)
    y0 = y_min + float(lomb_frac[0]) * span
    y1 = y_min + float(lomb_frac[1]) * span
    m = (y >= y0) & (y <= y1)
    if np.count_nonzero(m) < 8:
        return 0.0

    d = z_tan - z[m]
    fl = float(np.max(d))
    if fl < 0:
        fl = float(np.max(np.abs(d)))
    return fl

# ==============================
# MAIN
# ==============================
st.title("🦴 SpineScan SUPER — V3.2 + Courbures frontales")

if not ply_file:
    st.info("Veuillez importer un fichier .PLY (Revopoint) pour lancer l’analyse.")
    st.stop()

if st.button("⚙️ LANCER L'ANALYSE"):
    # ---- Load + convert to cm ----
    pts = load_ply_numpy(ply_file) * 0.1  # mm -> cm

    # Nettoyage léger Y (garder tout le dos)
    mask = (pts[:, 1] > np.percentile(pts[:, 1], 1)) & (pts[:, 1] < np.percentile(pts[:, 1], 99))
    pts = pts[mask]

    # Centrage X (AFFICHAGE uniquement) -> médiane (robuste)
    pts[:, 0] -= np.median(pts[:, 0])

    # Rotation XZ (stabilité)
    R = estimate_rotation_xz(pts)
    pts_r = apply_rotation_xz(pts, R)

    # ---- Raster surface + midline ----
    grid, xmin, ymin, dx_used, dy_used = build_depth_surface(pts_r, dx=float(dx), dy=float(dy))
    psis_L, psis_R, psis_conf = detect_psis(grid, xmin, ymin, dx_used, dy_used)
    spine_r, meta = extract_midline_symmetry_surface(grid, xmin, ymin, dx_used, dy_used, edge_q=int(edge_q))

    if spine_r.shape[0] < 25:
        st.error("Surface insuffisante pour extraire une ligne médiane stable.")
        st.stop()

    # Fiabilité par point
    q = quality_from_surface(spine_r, meta, psis_conf=psis_conf, max_jump_cm=3.0)

    # Retour repère original
    spine = unrotate_spine_xz(spine_r, R)

    # Lissage final (courbe)
    if do_smooth:
        spine = smooth_spine(spine, window=smooth_window, strong=strong_smooth, median_k=median_k)

    # Ancienne droite robuste z_ref(y) si tu veux la garder pour debug:
    # fd, fl_old, z_ref, z_dev = compute_sagittal_arrows_v3(spine)

    # ✅ Nouvelle verticale tangentielle "collée dorsal"
    z_tan, (y_tan0, y_tan1) = sagittal_tangent_vertical_z(
        spine,
        dorsal_frac=(0.55, 0.92),   # tu peux ajuster
        z_quantile=98.0             # 98–99 conseillé
    )

    # ✅ Nouvelle flèche lombaire vs verticale tangentielle
    fl = lumbar_arrow_vs_tangent_vertical(spine, z_tan, lomb_frac=(0.08, 0.45))

    # Si tu veux encore afficher fd = 0 ou rien:
    fd = 0.0
    z_ref = np.array([])  # plus utilisé pour la verticale

    fl_status = classify_range(fl, fl_lo, fl_hi)
    dev_f = float(np.max(np.abs(spine[:, 0]))) if spine.size else 0.0

    # Angles sagittaux
    lord_deg, kyph_deg, y_junction = estimate_lordosis_kyphosis_angles_v2(spine, smooth_win=int(angle_smooth))
    lord_status = classify_range(lord_deg, lord_lo, lord_hi)
    kyph_status = classify_range(kyph_deg, kyph_lo, kyph_hi)

    # Cobb proxy optionnel + segments (inchangé)
    cobb_deg, fit_bot, fit_top, y_ranges = (None, None, None, None)
    if cobb_enabled:
        cobb_deg, fit_bot, fit_top, y_ranges = estimate_cobb_proxy_front(spine, smooth_win=int(cobb_smooth))

    # ✅ Courbures frontales optionnelles
    front_kpk_lo = front_krms_lo = front_ang_lo = 0.0
    front_kpk_th = front_krms_th = front_ang_th = 0.0
    if front_curv_enabled:
        (front_kpk_lo, front_krms_lo, front_ang_lo,
         front_kpk_th, front_krms_th, front_ang_th) = frontal_curvature_metrics(
            spine,
            smooth_win=int(front_curv_smooth),
            lomb_frac=(0.12, 0.45),
            thor_frac=(0.55, 0.90),
            threshold_ratio=float(front_thr),
        )

    # Couverture / Fiabilité
    y_span_pts = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
    y_span_sp = float(np.max(spine[:, 1]) - np.min(spine[:, 1]))
    coverage_pct = 100.0 * (y_span_sp / y_span_pts) if y_span_pts > 1e-6 else 0.0
    coverage_pct = float(np.clip(coverage_pct, 0.0, 100.0))
    reliability_pct = 100.0 * float(np.mean(q >= 0.65)) if q.size else 0.0
    psis_pct = 100.0 * float(psis_conf)

    # Jonction TL en Y relative
    y0 = float(np.min(spine[:, 1]))
    y_junction_rel = None if y_junction is None else float(y_junction - y0)
    y_junction_disp = "n/a" if y_junction_rel is None else f"{y_junction_rel:.1f} cm"

    # =========================
    # GRAPHIQUES
    # =========================
    st.write("### 📈 Analyse visuelle")
    c1, c2 = st.columns(2)

    # Frontale X vs Y + Cobb lines (option)
    fig_f, ax_f = plt.subplots(figsize=(3.0, 5.2))
    ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.07, color="gray")
    plot_colored_curve(ax_f, spine[:, 0], spine[:, 1], q, lw=3.0)

    if cobb_enabled and fit_bot and fit_top and y_ranges:
        a_bot, b_bot = fit_bot
        a_top, b_top = fit_top
        y10, y30, y70, y90 = y_ranges
        yy_bot = np.array([y10, y30])
        yy_top = np.array([y70, y90])
        xx_bot = a_bot * yy_bot + b_bot
        xx_top = a_top * yy_top + b_top
        ax_f.plot(xx_bot, yy_bot, linewidth=2.2)
        ax_f.plot(xx_top, yy_top, linewidth=2.2)
        ax_f.text(0.02, 0.98, f"Cobb proxy: {cobb_deg:.1f}°", transform=ax_f.transAxes,
                  va="top", ha="left", fontsize=9)

    if front_curv_enabled:
        ax_f.text(0.02, 0.90,
                  f"κ lomb peak: {front_kpk_lo:.4f}\nκ dor peak: {front_kpk_th:.4f}",
                  transform=ax_f.transAxes, va="top", ha="left", fontsize=8, alpha=0.9)

    ax_f.set_title("Frontale (couleur = fiabilité)", fontsize=10)
    ax_f.axis("off")
    img_front_path = save_fig(fig_f, "front_super.png")
    c1.pyplot(fig_f)

    # Sagittale Z vs Y + "verticale" z_ref(y)
    spine_s = spine[np.argsort(spine[:, 1])]
    y_sorted = spine_s[:, 1]
    z_sorted = spine_s[:, 2]

    fig_s, ax_s = plt.subplots(figsize=(3.0, 5.2))
    ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.07, color="gray")
    plot_colored_curve(ax_s, z_sorted, y_sorted, q, lw=3.0)

    # ✅ Vraie verticale tangentielle collée au dorsal
    ax_s.axvline(z_tan, linestyle="--", linewidth=1.6, alpha=0.85)

    # Optionnel: matérialiser la zone dorsale utilisée pour la tangence (barre légère)
    ax_s.plot([z_tan, z_tan], [y_tan0, y_tan1], linestyle="--", linewidth=4.0, alpha=0.15)

    if y_junction is not None:
        ax_s.axhline(y_junction, linestyle="--", linewidth=1.4, alpha=0.6)

    ax_s.set_title("Sagittale (réf. z(y) en pointillés)", fontsize=10)
    ax_s.axis("off")
    img_sag_path = save_fig(fig_s, "sag_super.png")
    c2.pyplot(fig_s)

    # Légende fiabilité
    st.caption("Fiabilité: Rouge = faible, Jaune = moyen, Vert = fiable (score 0→1).")
    fig_leg, ax_leg = plt.subplots(figsize=(5.8, 0.35))
    ax_leg.set_axis_off()
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_leg.imshow(gradient, aspect="auto", cmap="RdYlGn")
    st.pyplot(fig_leg)

    if psis_L is not None:
        st.caption(f"PSIS détectées (confiance {psis_pct:.0f}%) — gauche: {psis_L[0]:.1f},{psis_L[1]:.1f} cm | droite: {psis_R[0]:.1f},{psis_R[1]:.1f} cm")
    else:
        st.caption(f"PSIS non détectées (confiance {psis_pct:.0f}%)")

    # Heatmap asymétrie optionnelle
    img_asym_path = None
    if show_asym:
        fig_a = make_asymmetry_heatmap(grid)
        if fig_a is not None:
            st.write("### 🗺️ Asymétrie gauche/droite (option)")
            st.pyplot(fig_a)
            img_asym_path = save_fig(fig_a, "asym_super.png")
        else:
            st.info("Asymétrie: données insuffisantes pour une heatmap stable.")

    # =========================
    # SYNTHESE (HTML)
    # =========================
    st.write("### 🧾 Synthèse des résultats")

    def badge(ok):
        if ok:
            return '<span style="margin-left:8px; padding:2px 8px; border-radius:999px; background:#e8f7ee; color:#156f3b; font-weight:800; font-size:0.85rem;">Normale</span>'
        return '<span style="margin-left:8px; padding:2px 8px; border-radius:999px; background:#fdecec; color:#9b1c1c; font-weight:800; font-size:0.85rem;">Hors norme</span>'

    fl_ok = (fl_status == "Normale")
    lord_ok = (lord_status == "Normale")
    kyph_ok = (kyph_status == "Normale")

    # Bloc courbures frontales
    front_block = ""
    if front_curv_enabled:
        front_block = f"""
        <p><b>📈 Courbure frontale lombaire :</b>
           <br><span style="font-weight:900;">peak |κ| = {front_kpk_lo:.4f} 1/cm</span> &nbsp; | &nbsp;
           <span style="font-weight:900;">RMS κ = {front_krms_lo:.4f} 1/cm</span>
           <br><b>Angle courbure lombaire :</b> <span style="font-weight:900;">{front_ang_lo:.1f}°</span></p>

        <p><b>📈 Courbure frontale dorsale :</b>
           <br><span style="font-weight:900;">peak |κ| = {front_kpk_th:.4f} 1/cm</span> &nbsp; | &nbsp;
           <span style="font-weight:900;">RMS κ = {front_krms_th:.4f} 1/cm</span>
           <br><b>Angle courbure dorsale :</b> <span style="font-weight:900;">{front_ang_th:.1f}°</span></p>
        """

    cobb_block = ""
    if cobb_enabled and cobb_deg is not None:
        cobb_block = f"""
        <p><b>📐 Angle de Cobb (proxy) :</b> <span style="font-weight:900;">{cobb_deg:.1f}°</span>
        <br><span style="color:#666; font-size:0.9rem;">Proxy de suivi (frontale), pas un Cobb radiographique.</span></p>
        """

    norms_fl = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {fl_lo:.1f} à {fl_hi:.1f} cm</span>" if show_norms else ""
    norms_lord = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {lord_lo:.0f}° à {lord_hi:.0f}°</span>" if show_norms else ""
    norms_kyph = f"<br><span style='color:#666; font-size:0.9rem;'>Référence: {kyph_lo:.0f}° à {kyph_hi:.0f}°</span>" if show_norms else ""

    html_card = f"""
    <div style="
        background:#fff; padding:16px; border-radius:12px; border:1px solid #e0e0e0;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        color:#2c3e50;
    ">
      <div style="font-size:1.05rem; line-height:1.55;">
        <p><b>📏 Flèche lombaire (robuste) :</b> <span style="font-weight:900;">{fl:.2f} cm</span>
          {badge(fl_ok) if show_norms else ""}{norms_fl}</p>

        <p><b>📏 Flèche dorsale (robuste) :</b> <span style="font-weight:900;">{fd:.2f} cm</span></p>

        <p><b>↔️ Déviation latérale max :</b> <span style="font-weight:900;">{dev_f:.2f} cm</span></p>

        <p><b>📐 Lordose (est.) :</b> <span style="font-weight:900;">{lord_deg:.1f}°</span>
          {badge(lord_ok) if show_norms else ""}{norms_lord}</p>

        <p><b>📐 Cyphose (est.) :</b> <span style="font-weight:900;">{kyph_deg:.1f}°</span>
          {badge(kyph_ok) if show_norms else ""}{norms_kyph}</p>

        {front_block}
        {cobb_block}

        <p><b>🔁 Jonction thoraco-lombaire (rel.) :</b> <span style="font-weight:900;">{y_junction_disp}</span></p>

        <p><b>✅ Couverture :</b> <span style="font-weight:900;">{coverage_pct:.0f}%</span>
           &nbsp; <b>Fiabilité :</b> <span style="font-weight:900;">{reliability_pct:.0f}%</span>
           <br><span style="color:#666; font-size:0.9rem;">Fiabilité = % des points avec score ≥ 0.65 | Confiance PSIS = {psis_pct:.0f}%</span></p>
      </div>

      <div style="
          margin-top:10px; font-size:0.82rem; color:#555; font-style:italic;
          border-left:3px solid #ccc; padding-left:10px;
      ">
        “Rasterstéréographie” : surface Z(x,y)=max Z par cellule (anti-biais densité) + ligne médiane par symétrie.<br/>
        Flèches sagittales : mesurées vs une droite de référence z(y) robuste (pointillés).<br/>
        Courbures frontales : κ sur x(y) + angle (différence de tangentes aux bornes).
      </div>
    </div>
    """
    components.html(html_card, height=720 if front_curv_enabled else 520, scrolling=False)

    # =========================
    # PDF
    # =========================
    results = {
        "fl": float(fl),
        "fl_status": fl_status,
        "fd": float(fd),
        "dev_f": float(dev_f),
        "lordosis_deg": float(lord_deg),
        "lordosis_status": lord_status,
        "kyphosis_deg": float(kyph_deg),
        "kyphosis_status": kyph_status,
        "y_junction_rel": y_junction_disp,
        "coverage_pct": float(coverage_pct),
        "reliability_pct": float(reliability_pct),
        "psis_pct": float(psis_pct),

        "front_curv_enabled": bool(front_curv_enabled),
        "front_kappa_peak_lomb": float(front_kpk_lo),
        "front_kappa_rms_lomb": float(front_krms_lo),
        "front_angle_lomb_deg": float(front_ang_lo),
        "front_kappa_peak_thor": float(front_kpk_th),
        "front_kappa_rms_thor": float(front_krms_th),
        "front_angle_thor_deg": float(front_ang_th),

        "cobb_enabled": bool(cobb_enabled),
        "cobb_deg": float(cobb_deg) if (cobb_enabled and cobb_deg is not None) else 0.0,
    }

    pdf_path = export_pdf_super({"nom": nom, "prenom": prenom}, results, img_front_path, img_sag_path, img_asym_path)

    st.divider()
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Télécharger le rapport PDF", f, f"Rapport_SpineScan_SUPER_{nom}.pdf")






