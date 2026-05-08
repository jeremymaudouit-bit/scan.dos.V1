import streamlit as st
import pyrealsense2 as rs
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev

import tempfile

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# =========================================================
# CONFIG STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Analyse Posturale 3D",
    layout="wide"
)

st.title("Analyse morphologique 3D - RealSense")

st.write("Importer un fichier .bag RealSense")

# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Choisir un fichier BAG",
    type=["bag"]
)

# =========================================================
# UTILS
# =========================================================

def midpoint(a, b):
    return (np.array(a) + np.array(b)) / 2


# =========================================================
# PDF
# =========================================================

def generate_pdf(metrics, output_path):

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Rapport Analyse Morphologique 3D",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    for k, v in metrics.items():

        elements.append(
            Paragraph(
                f"<b>{k}</b> : {v}",
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1, 10))

    doc.build(elements)


# =========================================================
# ANALYSE BAG
# =========================================================

def analyse_bag(bag_path):

    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_device_from_file(
        bag_path,
        repeat_playback=False
    )

    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()

    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    pc = rs.pointcloud()

    results = []

    frame_id = 0

    while True:

        try:
            frames = pipeline.wait_for_frames()
        except:
            break

        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # =================================================
        # POINT CLOUD
        # =================================================

        points = pc.calculate(depth_frame)

        vertices = np.asanyarray(
            points.get_vertices()
        )

        vertices = vertices.view(
            np.float32
        ).reshape(-1, 3)

        vertices = vertices[
            np.isfinite(vertices).all(axis=1)
        ]

        # =================================================
        # BODY FILTER
        # =================================================

        body = vertices[
            (vertices[:, 2] > 0.5) &
            (vertices[:, 2] < 2.2) &
            (np.abs(vertices[:, 0]) < 0.5)
        ]

        if len(body) < 500:
            continue

        # =================================================
        # SORT VERTICAL
        # =================================================

        body = body[np.argsort(body[:, 1])]

        n = len(body)

        upper = body[:n // 3]
        middle = body[n // 3:2 * n // 3]
        lower = body[2 * n // 3:]

        # =================================================
        # EPAULES
        # =================================================

        shoulder_left = upper[np.argmin(upper[:, 0])]
        shoulder_right = upper[np.argmax(upper[:, 0])]

        shoulder_center = midpoint(
            shoulder_left,
            shoulder_right
        )

        # =================================================
        # BASSIN
        # =================================================

        hip_left = lower[np.argmin(lower[:, 0])]
        hip_right = lower[np.argmax(lower[:, 0])]

        pelvis_center = midpoint(
            hip_left,
            hip_right
        )

        # =================================================
        # THORAX
        # =================================================

        thorax_left = middle[np.argmin(middle[:, 0])]
        thorax_right = middle[np.argmax(middle[:, 0])]

        # =================================================
        # MESURES
        # =================================================

        shoulder_width = euclidean(
            shoulder_left,
            shoulder_right
        )

        pelvis_width = euclidean(
            hip_left,
            hip_right
        )

        thorax_width = euclidean(
            thorax_left,
            thorax_right
        )

        trunk_length = euclidean(
            shoulder_center,
            pelvis_center
        )

        forward_tilt = (
            shoulder_center[2]
            - pelvis_center[2]
        )

        lateral_tilt = (
            shoulder_center[1]
            - pelvis_center[1]
        )

        # =================================================
        # DOS
        # =================================================

        back_zone = body[
            np.abs(body[:, 0]) < 0.15
        ]

        back_relief = np.std(
            back_zone[:, 2]
        )

        # =================================================
        # JAMBES
        # =================================================

        legs = vertices[
            vertices[:, 1] < pelvis_center[1]
        ]

        if len(legs) > 0:

            left_leg = legs[
                np.argmin(legs[:, 0])
            ]

            right_leg = legs[
                np.argmax(legs[:, 0])
            ]

            leg_span = euclidean(
                left_leg,
                right_leg
            )

        else:
            leg_span = 0

        # =================================================
        # COLONNE
        # =================================================

        trunk_axis = np.linspace(
            shoulder_center,
            pelvis_center,
            20
        )

        try:

            coords = [
                trunk_axis[:, 0],
                trunk_axis[:, 1],
                trunk_axis[:, 2]
            ]

            tck, u = splprep(
                coords,
                s=0.02
            )

            u_fine = np.linspace(
                0,
                1,
                50
            )

            spine = np.array(
                splev(u_fine, tck)
            ).T

            spine_curvature = np.std(
                spine[:, 2]
            )

        except:
            spine_curvature = 0

        # =================================================
        # SAVE
        # =================================================

        results.append({

            "Frame": frame_id,

            "Largeur épaules (m)": round(
                shoulder_width,
                3
            ),

            "Largeur bassin (m)": round(
                pelvis_width,
                3
            ),

            "Largeur thorax (m)": round(
                thorax_width,
                3
            ),

            "Longueur tronc (m)": round(
                trunk_length,
                3
            ),

            "Inclinaison avant/arrière": round(
                forward_tilt,
                3
            ),

            "Inclinaison latérale": round(
                lateral_tilt,
                3
            ),

            "Relief dos": round(
                back_relief,
                4
            ),

            "Courbure colonne": round(
                spine_curvature,
                4
            ),

            "Écart jambes": round(
                leg_span,
                3
            )

        })

        frame_id += 1

    pipeline.stop()

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".bag"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        bag_path = tmp.name

    st.info("Analyse en cours...")

    df = analyse_bag(bag_path)

    st.success("Analyse terminée")

    st.subheader("Mesures")

    st.dataframe(df)

    # =====================================================
    # MOYENNES
    # =====================================================

    st.subheader("Résumé moyen")

    mean_metrics = df.mean(
        numeric_only=True
    )

    st.write(mean_metrics)

    # =====================================================
    # GRAPH
    # =====================================================

    st.subheader("Évolution")

    st.line_chart(df[[
        "Largeur épaules (m)",
        "Largeur bassin (m)",
        "Relief dos"
    ]])

    # =====================================================
    # PDF
    # =====================================================

    pdf_metrics = {
        k: round(v, 3)
        for k, v in mean_metrics.items()
    }

    pdf_path = "rapport_postural.pdf"

    generate_pdf(
        pdf_metrics,
        pdf_path
    )

    with open(pdf_path, "rb") as f:

        st.download_button(
            "Télécharger PDF",
            f,
            file_name="rapport_postural.pdf"
        )
