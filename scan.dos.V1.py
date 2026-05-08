import streamlit as st
                trunk_axis[:, 1],
                trunk_axis[:, 2]
            ], s=0.02)

            u_fine = np.linspace(0, 1, 50)

            spine = np.array(splev(u_fine, tck)).T

            spine_curvature = np.std(spine[:, 2])

        except:
            spine_curvature = 0

        metrics = {
            "Frame": frame_count,
            "Largeur épaules (m)": round(shoulder_width, 3),
            "Largeur bassin (m)": round(pelvis_width, 3),
            "Largeur thorax (m)": round(thorax_width, 3),
            "Longueur tronc (m)": round(trunk_length, 3),
            "Inclinaison avant/arrière": round(forward_tilt, 3),
            "Inclinaison latérale": round(lateral_tilt, 3),
            "Relief dos": round(back_relief, 4),
            "Courbure colonne": round(spine_curvature, 4),
            "Écart jambes": round(leg_span, 3)
        }

        metrics_list.append(metrics)

        frame_count += 1

    pipeline.stop()

    return pd.DataFrame(metrics_list)


# ======================================================
# MAIN
# ======================================================

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bag") as tmp:
        tmp.write(uploaded_file.read())
        bag_path = tmp.name

    st.info("Analyse du fichier en cours...")

    df = analyse_bag(bag_path)

    st.success("Analyse terminée")

    st.dataframe(df)

    st.subheader("Résumé moyen")

    mean_metrics = df.mean(numeric_only=True)

    st.write(mean_metrics)

    st.line_chart(df[[
        "Largeur épaules (m)",
        "Largeur bassin (m)",
        "Relief dos"
    ]])

    # ==================================================
    # PDF
    # ==================================================

    pdf_metrics = {
        k: round(v, 3)
        for k, v in mean_metrics.to_dict().items()
    }

    pdf_path = generate_pdf(pdf_metrics)

    with open(pdf_path, "rb") as f:

        st.download_button(
            "Télécharger rapport PDF",
            f,
            file_name="rapport_postural.pdf"
        )
