import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
from datetime import datetime
from plyfile import PlyData
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ==============================
# CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="Analyse Rachis 3D IA", layout="wide")
st.title("🦴 Analyse rachidienne 3D – Synthèse clinique")

# ==============================
# UTILS
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    vertex = plydata['vertex']
    pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    return pts

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

def render_projection(points_cm, spine_cm, mode="front", z_ref=None, height_ratio=4):
    """Rendu 2D frontale ou sagittale pour Streamlit"""
    fig, ax = plt.subplots(figsize=(4, height_ratio))
    if mode == "front":
        ax.scatter(points_cm[:, 0], points_cm[:, 1], s=1, alpha=0.15)
        ax.plot(spine_cm[:, 0], spine_cm[:, 1], color="red", linewidth=2)
        ax.set_title("Vue frontale")
        # Pas d'inversion Y
    if mode == "side":
        ax.scatter(points_cm[:, 2], points_cm[:, 1], s=1, alpha=0.15)
        ax.plot(spine_cm[:, 2], spine_cm[:, 1], color="red", linewidth=2)
        if z_ref is not None:
            ax.plot(z_ref, spine_cm[:, 1], "--", color="black", linewidth=2)
        ax.set_title("Vue sagittale")
    ax.set_aspect("equal")
    ax.grid(True)
    return fig

def save_projection_pdf(fig, filename):
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def export_pdf(results, img_front, img_side):
    tmp = tempfile.gettempdir()
    pdf_path = os.path.join(tmp, "rapport_rachis.pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    story = []

    story.append(Paragraph("<b>Rapport d'analyse rachidienne 3D</b>", styles["Title"]))
    story.append(Spacer(1,0.4*cm))

    for k,v in results.items():
        story.append(Paragraph(f"<b>{k}</b> : {v}", styles["Normal"]))

    story.append(Spacer(1,0.5*cm))
    story.append(PDFImage(img_front, width=12*cm, height=9*cm))
    story.append(Spacer(1,0.3*cm))
    story.append(PDFImage(img_side, width=12*cm, height=9*cm))
    doc.build(story)
    return pdf_path

# ==============================
# SIDEBAR
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
# BOUTON ANALYSE
# ==============================
if ply_file and st.button("⚙️ Lancer l'analyse"):
    pts = load_ply_numpy(ply_file)
    pts *= 0.1
    mask = (pts[:,1] > np.percentile(pts[:,1],5)) & (pts[:,1] < np.percentile(pts[:,1],95))
    pts = pts[mask]
    pts[:,0] -= pts[:,0].mean()
    pts[:,2] -= pts[:,2].mean()

    slices = np.linspace(pts[:,1].min(), pts[:,1].max(), 60)
    spine = []
    for i in range(len(slices)-1):
        sl = pts[(pts[:,1]>=slices[i]) & (pts[:,1]<slices[i+1])]
        if len(sl)==0: continue
        x_mean,x_std = sl[:,0].mean(), sl[:,0].std()
        sl = sl[(sl[:,0] > x_mean - k_std*x_std) & (sl[:,0] < x_mean + k_std*x_std)]
        if len(sl): spine.append([sl[:,0].mean(), sl[:,1].mean(), sl[:,2].mean()])
    spine = np.array(spine)
    spine = spine[np.argsort(spine[:,1])]

    if smooth and len(spine) > 7:
        win = min(len(spine)//2*2+1, smooth_level)
        spine[:,0] = savgol_filter(spine[:,0], win, 3)
        spine[:,2] = savgol_filter(spine[:,2], win, 3)

    x,y,z = spine.T
    cobb = compute_cobb_angle(x,y)
    frontal_dev = np.max(np.abs(x))
    sagittal_dev = np.max(np.abs(z))
    fleche_dorsale, fleche_lombaire, z_ref = compute_sagittal_arrows(spine)

    # ==============================
    # RENDER STREAMLIT
    # ==============================
    tmp = tempfile.gettempdir()
    img_front = os.path.join(tmp,"front.png")
    img_side = os.path.join(tmp,"side.png")

    fig_front = render_projection(pts, spine, "front", height_ratio=2.5)  # réduit hauteur
    fig_side = render_projection(pts, spine, "side", z_ref=z_ref, height_ratio=2)  # plus compacte
    fig_front.savefig(img_front, bbox_inches="tight")
    fig_side.savefig(img_side, bbox_inches="tight")

    col_front, col_side = st.columns([1,1])
    col_front.image(img_front, caption="Vue frontale", use_column_width=True, output_format="PNG")
    col_side.image(img_side, caption="Vue sagittale", use_column_width=True, output_format="PNG")

    # ==============================
    # RÉSULTATS
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

    pdf_path = export_pdf(results, img_front, img_side)
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Télécharger le rapport PDF", f, "rapport_rachis.pdf")
