import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from scipy.signal import savgol_filter
import tempfile, os
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ==============================
# CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="Analyse Rachis 3D IA", layout="wide")
st.title("🦴 Analyse rachidienne 3D – Synthèse clinique")

# ==============================
# OUTILS
# ==============================
def compute_cobb_angle(x, y):
    dy = np.gradient(y)
    dx = np.gradient(x)
    slopes = dx / (dy + 1e-6)
    a1 = np.arctan(slopes.max())
    a2 = np.arctan(slopes.min())
    return np.degrees(abs(a1 - a2))


def compute_sagittal_arrows(spine_cm):
    y = spine_cm[:, 1]
    z = spine_cm[:, 2]
    z_ref = np.linspace(z[0], z[-1], len(z))
    delta = z - z_ref
    fleche_dorsale = abs(np.min(delta))
    fleche_lombaire = abs(np.max(delta))
    return fleche_dorsale, fleche_lombaire, z_ref


def render_projection(points_cm, spine_cm, mode="front", z_ref=None):
    fig, ax = plt.subplots(figsize=(5, 7))

    if mode == "front":
        ax.scatter(points_cm[:, 0], points_cm[:, 1], s=1, alpha=0.15)
        ax.plot(spine_cm[:, 0], spine_cm[:, 1], color="red", linewidth=2)
        ax.set_title("Vue frontale")

    if mode == "side":
        ax.scatter(points_cm[:, 2], points_cm[:, 1], s=1, alpha=0.15)
        ax.plot(spine_cm[:, 2], spine_cm[:, 1], color="red", linewidth=2)
        if z_ref is not None:
            ax.plot(z_ref, spine_cm[:, 1], "--", color="black", linewidth=2)
        ax.set_title("Vue sagittale")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True)
    return fig


def export_pdf(results, img_front, img_side):
    tmp = tempfile.gettempdir()
    pdf_path = os.path.join(tmp, "rapport_rachis.pdf")

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    story = []

    story.append(Paragraph("<b>Rapport d'analyse rachidienne 3D</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    for k, v in results.items():
        story.append(Paragraph(f"<b>{k}</b> : {v}", styles["Normal"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Image(img_front, width=7 * cm, height=10 * cm))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Image(img_side, width=7 * cm, height=10 * cm))

    doc.build(story)
    return pdf_path

# ==============================
# SIDEBAR – PARAMÈTRES
# ==============================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "Anonyme")
    prenom = st.text_input("Prénom", "")
    st.divider()

    smooth = st.checkbox("Activer le lissage", True)
    smooth_level = st.slider("Intensité lissage", 5, 40, 20)
    k_std = st.slider("Tolérance axe (K × std)", 0.5, 3.0, 1.5)

    st.divider()
    ply_file = st.file_uploader("📂 Charger un scan PLY", type=["ply"])

# ==============================
# TRAITEMENT
# ==============================
if ply_file:
    pts = np.loadtxt(ply_file, skiprows=10)[:, :3]  # fallback simple PLY ascii
    pts *= 0.1  # mm → cm

    # Nettoyage Y
    y_vals = pts[:, 1]
    mask = (y_vals > np.percentile(y_vals, 5)) & (y_vals < np.percentile(y_vals, 95))
    pts = pts[mask]

    # Centrage
    pts[:, 0] -= pts[:, 0].mean()
    pts[:, 2] -= pts[:, 2].mean()

    # Extraction axe
    slices = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 60)
    spine = []
    for i in range(len(slices) - 1):
        sl = pts[(pts[:, 1] >= slices[i]) & (pts[:, 1] < slices[i + 1])]
        if len(sl) == 0:
            continue
        x_mean, x_std = sl[:, 0].mean(), sl[:, 0].std()
        sl = sl[(sl[:, 0] > x_mean - k_std * x_std) & (sl[:, 0] < x_mean + k_std * x_std)]
        spine.append([sl[:, 0].mean(), sl[:, 1].mean(), sl[:, 2].mean()])

    spine = np.array(spine)
    spine = spine[np.argsort(spine[:, 1])]

    if smooth and len(spine) > 7:
        win = min(len(spine) // 2 * 2 + 1, smooth_level)
        spine[:, 0] = savgol_filter(spine[:, 0], win, 3)
        spine[:, 2] = savgol_filter(spine[:, 2], win, 3)

    # ==============================
    # MESURES
    # ==============================
    x, y, z = spine.T
    cobb = compute_cobb_angle(x, y)
    frontal_dev = np.max(np.abs(x))
    sagittal_dev = np.max(np.abs(z))
    fleche_dorsale, fleche_lombaire, z_ref = compute_sagittal_arrows(spine)

    # ==============================
    # VISU
    # ==============================
    fig_front = render_projection(pts, spine, "front")
    fig_side = render_projection(pts, spine, "side", z_ref=z_ref)

    tmp = tempfile.gettempdir()
    img_front = os.path.join(tmp, "front.png")
    img_side = os.path.join(tmp, "side.png")
    fig_front.savefig(img_front, bbox_inches="tight")
    fig_side.savefig(img_side, bbox_inches="tight")

    col1, col2 = st.columns(2)
    col1.pyplot(fig_front)
    col2.pyplot(fig_side)

    # ==============================
    # SYNTHÈSE
    # ==============================
    results = {
        "Patient": f"{prenom} {nom}",
        "Date": datetime.now().strftime("%d/%m/%Y"),
        "Angle de Cobb": f"{cobb:.1f} °",
        "Déviation frontale max": f"{frontal_dev:.2f} cm",
        "Déviation sagittale max": f"{sagittal_dev:.2f} cm",
        "Flèche dorsale": f"{fleche_dorsale:.2f} cm",
        "Flèche lombaire": f"{fleche_lombaire:.2f} cm",
    }

    st.subheader("📋 Synthèse clinique")
    st.table(results)

    # ==============================
    # PDF
    # ==============================
    pdf_path = export_pdf(results, img_front, img_side)
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Télécharger le rapport PDF", f, "rapport_rachis.pdf")
