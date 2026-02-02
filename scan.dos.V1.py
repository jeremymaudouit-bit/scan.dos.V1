import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import tempfile
from datetime import datetime
from plyfile import PlyData

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm as cm_unit

# ================= STREAMLIT CONFIG =================
st.set_page_config(page_title="Analyse Rachidienne 3D", layout="wide")
st.title("🦴 Analyse Rachidienne 3D – Cloud Safe")
st.markdown("---")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "Anonyme")
    prenom = st.text_input("Prénom", "")

    st.divider()
    st.header("⚙️ Paramètres")
    smooth = st.checkbox("Activer le lissage", True)
    smooth_level = st.slider("Niveau lissage", 5, 51, 31, step=2)
    k_std = st.slider("Tolérance filtrage (K×std)", 0.5, 3.0, 1.5, 0.1)

# ================= OUTILS =================
def load_ply_numpy(file):
    ply = PlyData.read(file)
    v = ply["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T

def compute_cobb(x, y):
    dy = np.gradient(y)
    dx = np.gradient(x)
    slopes = dx / (dy + 1e-6)
    return np.degrees(abs(np.arctan(slopes.max()) - np.arctan(slopes.min())))

def generate_pdf(data, img):
    filename = f"Rapport_Rachis_{data['Nom']}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)
    story = []

    story.append(Paragraph("<b>Rapport d’analyse rachidienne</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm_unit))

    for k, v in data.items():
        story.append(Paragraph(f"<b>{k} :</b> {v}", styles["Normal"]))

    story.append(Spacer(1, 0.4 * cm_unit))
    story.append(Image(img, width=16 * cm_unit, height=8 * cm_unit))

    doc.build(story)
    return filename

# ================= UPLOAD =================
uploaded = st.file_uploader("📁 Charger un fichier PLY", type="ply")

if uploaded:
    pts = load_ply_numpy(uploaded)

    if st.button("⚙️ LANCER L'ANALYSE", use_container_width=True):
        with st.spinner("Analyse de la colonne vertébrale..."):

            # Nettoyage
            y = pts[:, 1]
            mask = (y > np.percentile(y, 5)) & (y < np.percentile(y, 95))
            pts = pts[mask]

            pts[:, 0] -= pts[:, 0].mean()
            pts[:, 2] -= pts[:, 2].mean()

            # Extraction axe
            slices = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 50)
            spine = []

            for i in range(len(slices) - 1):
                sl = pts[(pts[:, 1] >= slices[i]) & (pts[:, 1] < slices[i + 1])]
                if len(sl) == 0:
                    continue
                xm, xs = sl[:, 0].mean(), sl[:, 0].std()
                sl = sl[(sl[:, 0] > xm - k_std * xs) & (sl[:, 0] < xm + k_std * xs)]
                if len(sl):
                    spine.append(sl.mean(axis=0))

            spine = np.array(spine)
            spine = spine[np.argsort(spine[:, 1])]

            if smooth and len(spine) > 7:
                spine[:, 0] = savgol_filter(spine[:, 0], smooth_level, 3)
                spine[:, 2] = savgol_filter(spine[:, 2], smooth_level, 3)

            spine_cm = spine / 10
            x, y, z = spine_cm.T

            cobb = compute_cobb(x, y)
            frontal = np.max(np.abs(x))
            sagittal = np.max(np.abs(z))

            # Graphique
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(x, y)
            ax[0].set_title(f"Vue frontale – Cobb {cobb:.1f}°")
            ax[0].set_aspect("equal")
            ax[0].grid()

            ax[1].plot(z, y)
            ax[1].set_title("Vue sagittale")
            ax[1].set_aspect("equal")
            ax[1].grid()

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmp.name)
            st.pyplot(fig)

            results = {
                "Nom": nom,
                "Prénom": prenom,
                "Angle de Cobb": f"{cobb:.2f}°",
                "Déviation frontale max": f"{frontal:.2f} cm",
                "Déviation sagittale max": f"{sagittal:.2f} cm"
            }

            st.subheader("📊 Résultats")
            st.table(results)

            pdf = generate_pdf(results, tmp.name)
            with open(pdf, "rb") as f:
                st.download_button("📥 Télécharger le PDF", f, pdf, "application/pdf")
