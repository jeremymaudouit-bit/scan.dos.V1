import streamlit as st
import numpy as np
import os
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import open3d as o3d

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from scipy.signal import savgol_filter
import tempfile
import os
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

# ================= CONFIG STREAMLIT =================
st.set_page_config(page_title="Analyse Rachidienne 3D", layout="wide")

st.title("🦴 Analyse Rachidienne 3D IA")
st.markdown("---")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom", value="Anonyme")
    prenom = st.text_input("Prénom", value="")
    st.divider()

    st.header("⚙️ Paramètres d'analyse")
    smooth_enabled = st.checkbox("Activer le lissage", value=True)
    smooth_level = st.slider("Niveau de lissage", 3, 50, 30)
    k_std = st.slider("Tolérance filtrage frontal (K×std)", 0.5, 3.0, 1.5, 0.1)

# ================= OUTILS =================
def compute_cobb_angle(x, y):
    dy = np.gradient(y)
    dx = np.gradient(x)
    slopes = dx / (dy + 1e-6)
    a1 = np.arctan(slopes.max())
    a2 = np.arctan(slopes.min())
    return np.degrees(abs(a1 - a2))

def capture_3d_image(pcd, spine, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    lines = [[i, i + 1] for i in range(len(spine) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(spine),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
    vis.add_geometry(line_set)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

def generate_pdf(data, img_graphes, img_3d):
    filename = f"Rapport_Rachis_{data['Nom']}.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Rapport d'analyse rachidienne 3D</b>", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(f"<b>Patient :</b> {data['Prenom']} {data['Nom']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    for k, v in data.items():
        if k not in ["Nom", "Prenom"]:
            story.append(Paragraph(f"<b>{k} :</b> {v}", styles["Normal"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Image(img_graphes, width=16 * cm, height=7 * cm))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Image(img_3d, width=16 * cm, height=10 * cm))

    doc.build(story)
    return filename

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📁 Charger un fichier PLY", type=["ply"])

if uploaded_file:
    tmp_ply = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    tmp_ply.write(uploaded_file.read())
    tmp_ply.close()

    if st.button("⚙️ LANCER L'ANALYSE RACHIDIENNE", use_container_width=True):
        with st.spinner("Analyse 3D de la colonne vertébrale..."):

            pcd = o3d.io.read_point_cloud(tmp_ply.name)
            pts = np.asarray(pcd.points)

            # Nettoyage Y
            y_vals = pts[:, 1]
            mask = (y_vals > np.percentile(y_vals, 5)) & (y_vals < np.percentile(y_vals, 95))
            pts = pts[mask]

            # Centrage
            pts[:, 0] -= pts[:, 0].mean()
            pts[:, 2] -= pts[:, 2].mean()

            # Couleurs profondeur
            z_norm = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-6)
            pcd.colors = o3d.utility.Vector3dVector(mcm.viridis(z_norm)[:, :3])

            # Extraction axe central
            slices = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 50)
            spine = []

            for i in range(len(slices) - 1):
                sl = pts[(pts[:, 1] >= slices[i]) & (pts[:, 1] < slices[i + 1])]
                if len(sl) == 0:
                    continue
                x_mean, x_std = sl[:, 0].mean(), sl[:, 0].std()
                mask_x = (sl[:, 0] > x_mean - k_std * x_std) & (sl[:, 0] < x_mean + k_std * x_std)
                sl = sl[mask_x]
                if len(sl) == 0:
                    continue
                spine.append([sl[:, 0].mean(), sl[:, 1].mean(), sl[:, 2].mean()])

            spine = np.array(spine)
            spine = spine[np.argsort(spine[:, 1])]

            # Lissage
            if smooth_enabled and len(spine) > 5:
                window = min(len(spine) // 2 * 2 + 1, smooth_level * 2 + 1)
                spine[:, 0] = savgol_filter(spine[:, 0], window, 3)
                spine[:, 2] = savgol_filter(spine[:, 2], window, 3)

            spine_cm = spine / 10
            x, y, z = spine_cm.T

            cobb = compute_cobb_angle(x, y)
            frontal_dev = np.max(np.abs(x))
            sagittal_dev = np.max(np.abs(z))

            # Graphiques
            tmpdir = tempfile.gettempdir()
            img_graphes = os.path.join(tmpdir, "graphes.png")
            img_3d = os.path.join(tmpdir, "spine3d.png")

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].plot(x, y, linewidth=2)
            axs[0].set_title(f"Vue frontale – Cobb {cobb:.1f}°")
            axs[0].set_aspect("equal")
            axs[0].grid(True)

            axs[1].plot(z, y, linewidth=2)
            axs[1].set_title("Vue sagittale")
            axs[1].set_aspect("equal")
            axs[1].grid(True)

            plt.tight_layout()
            plt.savefig(img_graphes)
            st.pyplot(fig)

            capture_3d_image(pcd, spine, img_3d)
            st.image(img_3d, caption="Reconstruction 3D rachidienne")

            results = {
                "Nom": nom,
                "Prenom": prenom,
                "Angle de Cobb": f"{cobb:.2f}°",
                "Déviation frontale max": f"{frontal_dev:.2f} cm",
                "Déviation sagittale max": f"{sagittal_dev:.2f} cm",
                "Lissage activé": smooth_enabled,
                "Niveau lissage": smooth_level,
                "Tolérance K": k_std
            }

            st.subheader("📊 Résultats cliniques")
            st.table(results)

            pdf_path = generate_pdf(results, img_graphes, img_3d)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Télécharger le rapport PDF",
                    f,
                    file_name=pdf_path,
                    mime="application/pdf",
                    use_container_width=True
                )

