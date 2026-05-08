import streamlit as st
import pyrealsense2 as rs

st.set_page_config(page_title="Test BAG RealSense", layout="wide")

st.title("Chargement fichier .bag RealSense")

uploaded_file = st.file_uploader("Importer un fichier .bag", type=["bag"])

if uploaded_file:

    st.success("Fichier chargé dans Streamlit")

    # Sauvegarde temporaire
    bag_path = "temp.bag"
    with open(bag_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Lecture du fichier en cours...")

    # =========================
    # OUVERTURE BAG
    # =========================

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(bag_path, repeat_playback=False)

    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # =========================
    # INFOS CAMÉRA
    # =========================

    device = profile.get_device()

    st.subheader("Informations appareil")

    st.write("Nom :", device.get_info(rs.camera_info.name))
    st.write("S/N :", device.get_info(rs.camera_info.serial_number))
    st.write("Firmware :", device.get_info(rs.camera_info.firmware_version))

    # =========================
    # TEST FRAMES
    # =========================

    st.subheader("Test frames")

    frame_count = 0

    try:
        while frame_count < 30:

            frames = pipeline.wait_for_frames()

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()

            if depth and color:
                frame_count += 1

        st.success(f"{frame_count} frames lues avec succès")

    except Exception as e:
        st.error(f"Erreur lecture BAG : {e}")

    pipeline.stop()
