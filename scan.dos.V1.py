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
# Metrics (sagittal unchanged)
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

# =========================================================
# ✅ FRONTAL FROM SCRATCH: "MOYENNE DE SURFACE" (pas points)
# =========================================================
def build_dorsal_surface_grid(pts_cm, cell_cm=0.3, z_percentile=95):
    """
    Construit une surface du dos en grille (x,y)->z_surface.
    Chaque cellule (x,y) a 1 valeur => poids surface uniforme (anti densité).

    Retourne:
      x_centers (nx,), y_centers (ny,), Z (ny,nx) avec np.nan si vide
    """
    x = pts_cm[:, 0]
    y = pts_cm[:, 1]
    z = pts_cm[:, 2]

    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])

    if xmax - xmin < 1e-6 or ymax - ymin < 1e-6:
        return None

    nx = max(20, int(np.ceil((xmax - xmin) / cell_cm)))
    ny = max(40, int(np.ceil((ymax - ymin) / cell_cm)))

    # indices de cellule
    ix = np.clip(((x - xmin) / cell_cm).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / cell_cm).astype(int), 0, ny - 1)

    # regrouper par (iy, ix) via clé
    key = iy * nx + ix
    order = np.argsort(key)
    key_s = key[order]
    z_s = z[order]

    # init grille
    Z = np.full((ny, nx), np.nan, dtype=float)

    # parcours groupes
    splits = np.where(np.diff(key_s) != 0)[0] + 1
    groups = np.split(z_s, splits)
    keys_g = np.split(key_s, splits)

    for k_arr, z_arr in zip(keys_g, groups):
        k0 = int(k_arr[0])
        gy = k0 // nx
        gx = k0 % nx
        # valeur de surface (dos): percentile élevé de z dans la cellule
        Z[gy, gx] = float(np.percentile(z_arr, z_percentile))

    x_centers = xmin + (np.arange(nx) + 0.5) * cell_cm
    y_centers = ymin + (np.arange(ny) + 0.5) * cell_cm
    return x_centers, y_centers, Z

def symmetry_center_1d(xc, zc):
    """
    Trouve le centre c qui rend z(x) le plus symétrique.
    """
    if len(xc) < 10:
        return float(np.median(xc))

    xmin, xmax = float(xc.min()), float(xc.max())
    span = xmax - xmin
    a = xmin + 0.10 * span
    b = xmax - 0.10 * span
    if b <= a:
        return float(np.median(xc))

    candidates = np.linspace(a, b, 31)
    best_c = None
    best_cost = np.inf

    for c in candidates:
        umax = min(c - xmin, xmax - c)
        if umax <= 0:
            continue
        us = np.linspace(0, umax, 25)
        zL = np.interp(c - us, xc, zc)
        zR = np.interp(c + us, xc, zc)
        cost = float(np.mean(np.abs(zR - zL)))
        if cost < best_cost:
            best_cost = cost
            best_c = c

    return float(best_c if best_c is not None else np.median(xc))

def frontal_centerline_from_surface(pts_cm, cell_cm=0.3, z_cell_percentile=95,
                                    keep_top_percent_rows=35):
    """
    Construit la ligne frontale X(Y) à partir d'une SURFACE (grille).
    - chaque cellule pèse pareil (anti densité)
    - pour chaque Y, on utilise le profil z(x) et on prend le centre par symétrie

    keep_top_percent_rows: on garde seulement les cellules les plus dorsales dans la ligne Y
    (utile si la grille contient du bruit "devant" / trous).
    """
    grid = build_dorsal_surface_grid(pts_cm, cell_cm=cell_cm, z_percentile=z_cell_percentile)
    if grid is None:
        return np.empty((0, 3), dtype=float), None

    xc, yc, Z = grid
    ny, nx = Z.shape

    spine = []
    for j in range(ny):
        row = Z[j, :]
        ok = np.isfinite(row)
        if np.count_nonzero(ok) < max(8, nx // 10):
            continue

        x_row = xc[ok]
        z_row = row[ok]

        # option: ne garder que le "dos" dans cette ligne (top z)
        thr = np.percentile(z_row, 100 - keep_top_percent_rows)
        sel = z_row >= thr
        if np.count_nonzero(sel) >= 8:
            x_row = x_row[sel]
            z_row = z_row[sel]

        # léger lissage du profil (si possible)
        if len(z_row) >= 11:
            # ordonner par x
            order = np.argsort(x_row)
            x_row = x_row[order]
            z_row = z_row[order]
            w = 9 if len(z_row) >= 9 else (len(z_row) // 2) * 2 + 1
            if w >= 5 and w < len(z_row):
                z_row = savgol_filter(z_row, w, 2)

        x0 = symmetry_center_1d(x_row, z_row)
        y0 = float(yc[j])

        # pour z0 (sagittal de la ligne), on peut prendre la moyenne "surface" dorsale de la rangée
        z0 = float(np.nanpercentile(row, 90))

        spine.append([x0, y0, z0])

    spine = np.array(spine, dtype=float)
    if spine.shape[0] == 0:
        return spine, grid
    spine = spine[np.argsort(spine[:, 1])]
    return spine, grid

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

    st.subheader("🧱 Surface (plan frontal)")
    cell_mm = st.slider("Taille cellule surface (mm)", 2, 10, 4)  # 4mm par défaut
    z_cell_percentile = st.slider("Surface dos (percentile z/cellule)", 85, 99, 95)
    keep_top_rows = st.slider("Garder dos (top % par ligne y)", 10, 60, 35)

    st.divider()
    ply_file = st.file_uploader("Charger Scan (.PLY)", type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE"):
        # --- LOAD ---
        pts = load_ply_numpy(ply_file)

        # IMPORTANT: tu faisais *0.1 (mm->cm).
        # Ici je garde ton hypothèse "PLY en mm". Si ton PLY est déjà en cm, retire *0.1.
        pts = pts * 0.1

        # nettoyage Y (comme avant)
        mask = (pts[:, 1] > np.percentile(pts[:, 1], 5)) & (pts[:, 1] < np.percentile(pts[:, 1], 95))
        pts = pts[mask]

        # centrage X robuste (utile)
        pts[:, 0] -= np.median(pts[:, 0])

        # --- FRONTAL: surface-weight extraction ---
        cell_cm = float(cell_mm) / 10.0
        spine, grid = frontal_centerline_from_surface(
            pts_cm=pts,
            cell_cm=cell_cm,
            z_cell_percentile=z_cell_percentile,
            keep_top_percent_rows=keep_top_rows
        )

        if spine.shape[0] < 12:
            st.error(
                "Extraction insuffisante dans le frontal. "
                "Essaie: augmenter la taille cellule (mm), ou baisser 'top % par ligne', "
                "ou vérifier l'échelle (*0.1)."
            )
            st.stop()

        # Lissage
        if do_smooth and spine.shape[0] > smooth_val:
            spine = smooth_spine(spine, smooth_val=smooth_val, poly=3)

        # --- METRICS (sagittal à partir de z0 de la surface) ---
        fd, fl, vertical_z = compute_sagittal_arrow_lombaire_v2(spine)
        dev_f = float(np.max(np.abs(spine[:, 0])))

        # --- PLOTS ---
        tmp = tempfile.gettempdir()
        img_f_p, img_s_p = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")

        fig_f, ax_f = plt.subplots(figsize=(2.2, 4))
        ax_f.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_f.plot(spine[:, 0], spine[:, 1], "red", linewidth=2.0)
        ax_f.set_title("Frontale (moyenne surface)", fontsize=9)
        ax_f.axis("off")
        fig_f.savefig(img_f_p, bbox_inches="tight", dpi=160)

        fig_s, ax_s = plt.subplots(figsize=(2.2, 4))
        ax_s.scatter(pts[:, 2], pts[:, 1], s=0.2, alpha=0.08, color="gray")
        ax_s.plot(spine[:, 2], spine[:, 1], "blue", linewidth=2.0)
        ax_s.plot(vertical_z, spine[:, 1], "k--", alpha=0.7, linewidth=1)
        ax_s.set_title("Sagittale", fontsize=9)
        ax_s.axis("off")
        fig_s.savefig(img_s_p, bbox_inches="tight", dpi=160)

        # --- DISPLAY ---
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
                Plan frontal: <b>moyenne de surface</b> via grille (x,y) → z_surface (percentile) donc pas d'influence du nombre de points.
                Axe: centre par <b>symétrie</b> du profil z(x) à chaque hauteur y.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- PDF ---
        res = {"fd": fd, "fl": fl, "dev_f": dev_f}
        pdf_path = export_pdf_pro({"nom": nom, "prenom": prenom}, res, img_f_p, img_s_p)

        st.divider()
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Bilan_Spine_{nom}.pdf")
else:
    st.info("Veuillez importer un fichier .PLY pour lancer l'analyse.")
