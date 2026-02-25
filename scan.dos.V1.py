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
# ROBUST MIDLINE (SURFACE + SYMMETRY)
# =========================================================
def extract_midline(pts, remove_shoulders=True):

    y = pts[:,1]
    slices = np.linspace(np.percentile(y,8), np.percentile(y,92), 120)

    spine=[]
    prev_x=None

    for i in range(len(slices)-1):

        sl = pts[(y>=slices[i])&(y<slices[i+1])]
        if len(sl)<30:
            continue

        xvals = sl[:,0]
        zvals = sl[:,2]

        if remove_shoulders:
            thr = np.percentile(zvals,80)
            mask = zvals>=thr
            if np.sum(mask)>10:
                xvals = xvals[mask]

        x0 = np.median(xvals)
        y0 = np.mean(sl[:,1])
        z0 = np.percentile(zvals,90)

        if prev_x is not None and abs(x0-prev_x)>1.5:
            x0=prev_x

        prev_x=x0
        spine.append([x0,y0,z0])

    if len(spine)==0:
        return np.empty((0,3))

    return np.array(spine)

# =========================================================
# LISSAGE FORT
# =========================================================
def smooth_spine(spine):
    if len(spine)<11:
        return spine
    w=min(101,len(spine)-1)
    if w%2==0:
        w-=1
    spine[:,0]=savgol_filter(spine[:,0],w,3)
    spine[:,2]=savgol_filter(spine[:,2],w,3)
    return spine

# =========================================================
# SAGITTAL METRICS
# =========================================================
def sagittal_metrics(spine):
    if len(spine)==0:
        return 0,0,np.array([])
    z=spine[:,2]
    fl=float(abs(np.max(z)-np.min(z)))
    vertical=np.full_like(z,np.max(z))
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
    nom=st.text_input("Nom","DURAND")
    prenom=st.text_input("Prénom","Jean")
    remove_shoulders=st.toggle("Supprimer épaules",True)
    smooth_on=st.toggle("Lissage fort",True)
    ply_file=st.file_uploader("Charger Scan (.PLY)",type=["ply"])

st.title("🦴 SpineScan Pro")

# =========================================================
# MAIN
# =========================================================
if ply_file:
    if st.button("LANCER ANALYSE"):

        pts=load_ply_numpy(ply_file)*0.1

        # centrage x
        pts[:,0]-=np.median(pts[:,0])

        spine=extract_midline(pts,remove_shoulders)

        if len(spine)==0:
            st.error("Extraction impossible : scan insuffisant.")
            st.stop()

        if smooth_on:
            spine=smooth_spine(spine)

        fd,fl,vertical=sagittal_metrics(spine)

        tmp=tempfile.gettempdir()
        img_f=os.path.join(tmp,"front.png")
        img_s=os.path.join(tmp,"sag.png")

        # FRONTAL
        fig,ax=plt.subplots(figsize=(3,5))
        ax.scatter(pts[:,0],pts[:,1],s=0.2,alpha=0.08)
        ax.plot(spine[:,0],spine[:,1],'r',linewidth=2)
        ax.set_title("Frontale")
        ax.axis("off")
        fig.savefig(img_f,dpi=150)

        # SAGITTAL
        fig2,ax2=plt.subplots(figsize=(3,5))
        ax2.scatter(pts[:,2],pts[:,1],s=0.2,alpha=0.08)
        ax2.plot(spine[:,2],spine[:,1],'b',linewidth=2)
        ax2.plot(vertical,spine[:,1],'k--')
        ax2.set_title("Sagittale")
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
