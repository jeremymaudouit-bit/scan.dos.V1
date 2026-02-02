import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
from datetime import datetime
from plyfile import PlyData
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# ==============================
# CONFIG & THÈME
# ==============================
st.set_page_config(page_title="SpineScan Pro - 3D Analysis", layout="wide")

# CSS pour un look "Médical Tech"
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOGIQUE DE CALCUL (Optimisée)
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    vertex = plydata['vertex']
    return np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

def compute_metrics(spine, z_ref):
    x, y, z = spine.T
    # Angle de Cobb simplifié (max pente)
    dy, dx = np.gradient(y), np.gradient(x)
    slopes = dx / (dy + 1e-6)
    cobb = np.degrees(abs(np.arctan(slopes.max()) - np.arctan(slopes.min())))
    
    delta = z - z_ref
    f_dorsale = abs(np.max(delta))
    f_lombaire = abs(np.min(delta))
    
    return round(cobb, 1), round(f_dorsale, 2), round(f_lombaire, 2)

def get_status(cobb):
    if cobb < 10: return "Normal", colors.green
    if cobb < 20: return "Surveillance", colors.orange
    return "Avis Spécialisé", colors.red

# ==============================
# GÉNÉRATION PDF PRO
# ==============================
def export_pdf_pro(patient_info, results, img_front, img_side):
    tmp = tempfile.gettempdir()
    pdf_path = os.path.join(tmp, "bilan_rachis_pro.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor("#1A3A5A"), spaceAfter=20)
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=12, textColor=colors.grey, spaceAfter=12)
    
    story = []

    # En-tête
    story.append(Paragraph("BILAN RADIOLOGIQUE NUMÉRIQUE 3D", title_style))
    story.append(Paragraph(f"Édité le : {datetime.now().strftime('%d/%m/%Y à %H:%M')}", subtitle_style))
    story.append(Spacer(1, 0.5*cm))

    # Tableau Infos Patient
    data_patient = [
        ["PATIENT", f"{patient_info['prenom']} {patient_info['nom']}"],
        ["IDENTIFIANT", f"ID-{datetime.now().strftime('%y%m%d')}"],
        ["PRATICIEN", "Analyse Système SpineScan"]
    ]
    t_patient = Table(data_patient, colWidths=[4*cm, 10*cm])
    t_patient.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke), ('PADDING', (0,0), (-1,-1), 6)]))
    story.append(t_patient)
    story.append(Spacer(1, 1*cm))

    # Résultats Clés
    story.append(Paragraph("Mesures Morphométriques", styles['Heading2']))
    status_text, status_color = get_status(results['cobb'])
    
    res_data = [
        ["Indicateur", "Valeur", "Interprétation"],
        ["Angle de Cobb", f"{results['cobb']}°", status_text],
        ["Flèche Dorsale", f"{results['fd']} cm", "-"],
        ["Flèche Lombaire", f"{results['fl']} cm", "-"]
    ]
    t_res = Table(res_data, colWidths=[6*cm, 4*cm, 5*cm])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A3A5A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('TEXTCOLOR', (2,1), (2,1), status_color)
    ]))
    story.append(t_res)
    story.append(Spacer(1, 1*cm))

    # Images
    story.append(Paragraph("Visualisation des Courbures", styles['Heading2']))
    img_table = Table([[PDFImage(img_front, width=7*cm, height=10*cm), PDFImage(img_side, width=7*cm, height=10*cm)]])
    story.append(img_table)

    doc.build(story)
    return pdf_path

# ==============================
# INTERFACE STREAMLIT
# ==============================
st.title("🦴 SpineScan Pro")
st.caption("Système d'aide au diagnostic rachidien par imagerie 3D")

with st.sidebar:
    st.header("📋 Informations Patient")
    nom = st.text_input("Nom de famille")
    prenom = st.text_input("Prénom")
    st.divider()
    ply_file = st.file_uploader("Charger le scan (.ply)", type=["ply"])
    analyze_btn = st.button("Lancer l'Analyse Biomécanique")

if ply_file and analyze_btn:
    with st.spinner("Analyse des nuages de points en cours..."):
        # Prétraitement
        pts = load_ply_numpy(ply_file) * 0.1 # mm to cm
        
        # Algorithme d'extraction (simplifié pour l'exemple)
        z_min, z_max = np.percentile(pts[:, 1], [5, 95])
        mask = (pts[:, 1] > z_min) & (pts[:, 1] < z_max)
        pts = pts[mask]
        
        # Simulation d'extraction de la ligne centrale
        slices = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 50)
        spine = np.array([[pts[(pts[:, 1] > s) & (pts[:, 1] < s+1)][:, 0].mean(), s, 0] for s in slices])
        spine[:, 0] = savgol_filter(spine[:, 0], 11, 3) # Lissage
        
        z_ref = np.zeros_like(slices)
        cobb, fd, fl = compute_metrics(spine, z_ref)

        # Affichage des métriques avec colonnes
        m1, m2, m3 = st.columns(3)
        m1.metric("Angle de Cobb", f"{cobb}°", delta="Pathologique" if cobb > 10 else "Normal", delta_color="inverse")
        m2.metric("Flèche Dorsale", f"{fd} cm")
        m3.metric("Flèche Lombaire", f"{fl} cm")

        # Plots
        col_img1, col_img2 = st.columns(2)
        
        tmp = tempfile.gettempdir()
        path_f, path_s = os.path.join(tmp, "f.png"), os.path.join(tmp, "s.png")
        
        # Plot Frontal
        fig_f, ax_f = plt.subplots(figsize=(3, 5))
        ax_f.plot(spine[:, 0], spine[:, 1], 'r-', linewidth=3)
        ax_f.set_title("Ligne de Gravité (Frontale)")
        fig_f.savefig(path_f)
        col_img1.pyplot(fig_f)
        
        # Plot Sagittal
        fig_s, ax_s = plt.subplots(figsize=(3, 5))
        ax_s.plot(np.sin(slices/10)*2, slices, 'b-', linewidth=3) # Simulé
        ax_s.set_title("Profil Sagittal")
        fig_s.savefig(path_s)
        col_img2.pyplot(fig_s)

        # Export PDF
        results = {"cobb": cobb, "fd": fd, "fl": fl}
        patient_info = {"nom": nom, "prenom": prenom}
        pdf_file = export_pdf_pro(patient_info, results, path_f, path_s)
        
        st.divider()
        with open(pdf_file, "rb") as f:
            st.download_button("📂 Télécharger le Bilan Médical PDF", f, file_name=f"Bilan_{nom}.pdf")
else:
    st.info("Veuillez charger un fichier .ply dans la barre latérale pour commencer.")
