import streamlit as st
import numpy as np
import pandas as pd
import tempfile
from scipy.spatial.distance import euclidean
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Analyse posture 3D", layout="wide")

st.title("Analyse morphologique 3D (FULL CLOUD SAFE)")

uploaded_file = st.file_uploader(
    "Importer un fichier CSV de points 3D",
    type=["csv"]
)

# =========================
# UTILS
# =========================

def midpoint(a, b):
    return (np.array(a) + np.array(b)) / 2

# =========================
# PDF
# =========================

def export_pdf(metrics, path):

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Rapport postural 3D", styles["Title"]))
    elements.append(Spacer(1, 20))

    for k, v in metrics.items():
        elements.append(Paragraph(f"<b>{k}</b> : {v}", styles["BodyText"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)

# =========================
# ANALYSE CSV POINT CLOUD
# =========================

def analyze_csv(file_path):

    df = pd.read_csv(file_path)

    points = df[["x", "y", "z"]].values

    body = points[
        (points[:, 2] > 0.5) &
        (points[:, 2] < 2.2) &
        (np.abs(points[:, 0]) < 0.5)
    ]

    body = body[np.argsort(body[:, 1])]

    n = len(body)

    upper = body[:n//3]
    lower = body[2*n//3:]

    # épaules
    sh_l = upper[np.argmin(upper[:, 0])]
    sh_r = upper[np.argmax(upper[:, 0])]
    sh_c = midpoint(sh_l, sh_r)

    # bassin
    hip_l = lower[np.argmin(lower[:, 0])]
    hip_r = lower[np.argmax(lower[:, 0])]
    hip_c = midpoint(hip_l, hip_r)

    # mesures
    shoulder_width = euclidean(sh_l, sh_r)
    pelvis_width = euclidean(hip_l, hip_r)
    trunk_length = euclidean(sh_c, hip_c)

    forward = sh_c[2] - hip_c[2]
    lateral = sh_c[1] - hip_c[1]

    back_zone = body[np.abs(body[:, 0]) < 0.15]
    back_relief = np.std(back_zone[:, 2])

    return {
        "Largeur épaules": shoulder_width,
        "Largeur bassin": pelvis_width,
        "Longueur tronc": trunk_length,
        "Inclinaison avant/arrière": forward,
        "Inclinaison latérale": lateral,
        "Relief dos": back_relief
    }

# =========================
# MAIN
# =========================

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    st.info("Analyse en cours...")

    metrics = analyze_csv(path)

    st.success("Terminé")

    st.write(metrics)

    pdf_path = "rapport.pdf"
    export_pdf(metrics, pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("Télécharger PDF", f, file_name="rapport.pdf")
