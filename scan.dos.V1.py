import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
from plyfile import PlyData
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

st.set_page_config(page_title="SpineScan Pro 3D", layout="wide")

# =========================================================
# LOAD
# =========================================================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

# =========================================================
# AUTO DETECT VERTICAL AXIS
# =========================================================
def detect_vertical_axis(pts):
    ranges = np.ptp(pts, axis=0)
    return np.argmax(ranges)

# =========================================================
# PCA PROJECTION CLINIQUE
# =========================================================
def project_clinical(pts):
    center = np.mean(pts, axis=0)
    pts_c = pts - center
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    basis = Vt[:2]
    proj = pts_c @ basis.T
    return proj

# =========================================================
# ROBUST MIDLINE EXTRACTION (NO GRID FAILURE)
# =========================================================
def extract_midline(pts, remove_shoulders=True):

    # auto vertical
    vertical_axis = detect_vertical_axis(pts)

    # reorder so Y = vertical
    order = [0,1,2]
    order[1], order[vertical_axis] = order[vertical_axis], order[1]
    pts = pts[:,order]

    # PCA projection
    proj = project_clinical(pts)

    X = proj[:,0]
    Y = pts[:,1]
    Z = proj[:,1]

    # adaptive slicing
    n_slices = 120
    ys = np.linspace(np.percentile(Y,5), np.percentile(Y,95), n_slices)

    spine = []
    prev_x = None

    for i in range(len(ys)-1):
        mask = (Y>=ys[i]) & (Y<ys[i+1])
        if np.sum(mask)<30:
            continue

        xvals = X[mask]
        zvals = Z[mask]

        if remove_shoulders:
            thr = np.percentile(zvals,80)
            sel = zvals>=thr
            if np.sum(sel)>10:
                xvals = xvals[sel]

        if len(xvals)<5:
            continue

        x0 = np.median(xvals)
        y0 = np.mean(Y[mask])
        z0 = np.percentile(zvals,90)

        if prev_x is not None and abs(x0-prev_x)>2:
            x0 = prev_x

        prev_x = x0
        spine.append([x0,y0,z0])

    if len(spine)<10:
        return np.empty((0,3))

    spine = np.array(spine)
    return spine

# =========================================================
# SMOOTH
# =========================================================
def smooth_spine(spine):
    if len(spine)<11:
        return spine
    w = min(101, len(spine)-1)
    if w%2==0:
        w-=1
    spine[:,0] = savgol_filter(spine[:,0], w, 3)
    spine[:,2] = savgol_filter(spine[:,2], w, 3)
    return spine

# =========================================================
# SAGITTAL METRICS
# =========================================================
def sagittal_metrics(spine):
    if len(spine)==0:
        return 0,0,np.array([])
    z = spine[:,2]
    fl = float(abs(np.max(z)-np.min(z)))
    vertical = np.full_like(z,np.max(z))
    return 0,fl,vertical

# =========================================================
# PDF
# =========================================================
def export_pdf(patient,results,img_f,img_s):
    tmp=tempfile.gettempdir()
    path=os.path.join(tmp,"report_spine.pdf")
    doc=SimpleDocTemplate(path,pagesize=A4)
    styles=getSampleStyleSheet()
    story=[]
    story.append(Paragraph("<b>Bilan Rachidien 3D</b>",styles["Heading1"]))
    story.append(Spacer(1,1*cm))
    story.append(Paragraph(patient,styles["Normal"]))
    story.append(Spacer(1,1*cm))
    story.append(Paragraph(f"Flèche lombaire: {results['fl']:.2f} cm",styles["Normal"]))
    story.append(Spacer(1,1*cm))
    story.append(PDFImage(img_f,6*cm,9*cm))
    story.append(PDFImage(img_s,6*cm,9*cm))
    doc.build(story)
    return path

# =========================================================
# UI
# =========================================================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    remove_shoulders = st.toggle("Supprimer épaules",True)
    smooth_on = st.toggle("Lissage fort",True)
    ply_file = st.file_uploader("Charger Scan (.PLY)",type=["ply"])

st.title("🦴 SpineScan Pro")

# =========================================================
# MAIN
# =========================================================
if ply_file:
    if st.button("LANCER ANALYSE"):

        pts = load_ply_numpy(ply_file)

        # auto scale detection (mm or cm)
        if np.max(pts)<10:
            pts = pts
        elif np.max(pts)>1000:
            pts = pts*0.001
        else:
            pts = pts*0.1

        spine = extract_midline(pts,remove_shoulders)

        if len(spine)==0:
            st.error("Impossible d'extraire la colonne — scan insuffisant.")
            st.stop()

        if smooth_on:
            spine = smooth_spine(spine)

        fd,fl,vertical = sagittal_metrics(spine)

        tmp=tempfile.gettempdir()
        img_f=os.path.join(tmp,"front.png")
        img_s=os.path.join(tmp,"sag.png")

        fig,ax=plt.subplots(figsize=(3,5))
        ax.scatter(spine[:,0],spine[:,1],s=5)
        ax.plot(spine[:,0],spine[:,1],'r',linewidth=2)
        ax.set_title("Projection Frontale Clinique")
        ax.axis("off")
        fig.savefig(img_f,dpi=150)

        fig2,ax2=plt.subplots(figsize=(3,5))
        ax2.plot(spine[:,2],spine[:,1],'b',linewidth=2)
        ax2.plot(vertical,spine[:,1],'k--')
        ax2.set_title("Projection Sagittale Clinique")
        ax2.axis("off")
        fig2.savefig(img_s,dpi=150)

        st.pyplot(fig)
        st.pyplot(fig2)

        st.markdown(f"### Flèche lombaire : {fl:.2f} cm")

        pdf=export_pdf(f"{prenom} {nom}",{"fl":fl},img_f,img_s)
        with open(pdf,"rb") as f:
            st.download_button("Télécharger PDF",f,"rapport_spine.pdf")

else:
    st.info("Importer un scan .PLY")
