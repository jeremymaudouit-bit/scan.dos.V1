import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import tempfile, os
from plyfile import PlyData
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="SpineScan Pro 3D", layout="wide")

# ==============================
# IO
# ==============================
def load_ply_numpy(file):
    plydata = PlyData.read(file)
    v = plydata["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(float)

# ==============================
# LISSAGE FORT
# ==============================
def smooth_spine(spine, window=61):
    if spine.shape[0] < 11:
        return spine
    w = min(window, spine.shape[0]-1)
    if w % 2 == 0:
        w -= 1
    spine[:,0] = savgol_filter(spine[:,0], w, 3)
    spine[:,1] = savgol_filter(spine[:,1], w, 3)
    spine[:,2] = savgol_filter(spine[:,2], w, 3)
    return spine

# ==============================
# PROJECTION 3D -> 2D CLINIQUE (PCA TRONC)
# ==============================
def project_to_clinical_plane(pts):
    center = np.mean(pts, axis=0)
    pts_c = pts - center
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    basis = Vt[:2]  # 2 axes principaux
    proj = pts_c @ basis.T
    return proj, basis, center

# ==============================
# SURFACE GRID (anti densité)
# ==============================
def build_surface_grid(pts, cell=0.4):
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    xmin,xmax = np.percentile(x,[2,98])
    ymin,ymax = np.percentile(y,[2,98])
    nx = int((xmax-xmin)/cell)
    ny = int((ymax-ymin)/cell)
    ix = ((x-xmin)/cell).astype(int)
    iy = ((y-ymin)/cell).astype(int)
    Z = np.full((ny,nx),np.nan)
    for i in range(len(pts)):
        if 0<=ix[i]<nx and 0<=iy[i]<ny:
            if np.isnan(Z[iy[i],ix[i]]):
                Z[iy[i],ix[i]] = z[i]
            else:
                Z[iy[i],ix[i]] = max(Z[iy[i],ix[i]], z[i])
    xc = xmin + (np.arange(nx)+0.5)*cell
    yc = ymin + (np.arange(ny)+0.5)*cell
    return xc,yc,Z

# ==============================
# AXE MÉDIAN ROBUSTE (PCA LOCALE)
# ==============================
def anatomical_midline(pts, remove_shoulders=True):
    xc,yc,Z = build_surface_grid(pts)
    spine=[]
    prev_x=None
    for j in range(len(yc)):
        row=Z[j,:]
        ok=~np.isnan(row)
        if np.sum(ok)<8:
            continue
        xvals=xc[ok]
        zvals=row[ok]

        # suppression épaules
        if remove_shoulders:
            thr=np.percentile(zvals,85)
            sel=zvals>=thr
            xvals=xvals[sel]
            zvals=zvals[sel]

        if len(xvals)<6:
            continue

        # PCA locale 2D
        data=np.vstack([xvals,zvals]).T
        data-=np.mean(data,axis=0)
        U,S,Vt=np.linalg.svd(data,full_matrices=False)
        axis=Vt[0]
        proj=data@axis
        center=np.mean(data[proj.argsort()][len(proj)//2])
        x0=np.median(xvals)

        if prev_x is not None and abs(x0-prev_x)>1.0:
            x0=prev_x
        prev_x=x0

        spine.append([x0,yc[j],np.percentile(zvals,90)])

    return np.array(spine)

# ==============================
# SAGITTAL METRICS
# ==============================
def sagittal_metrics(spine):
    z=spine[:,2]
    fd=0
    fl=abs(np.min(z)-np.max(z))
    vertical=np.full_like(z,np.max(z))
    return fd,fl,vertical

# ==============================
# PDF
# ==============================
def export_pdf(patient,results,img_f,img_s):
    tmp=tempfile.gettempdir()
    path=os.path.join(tmp,"report.pdf")
    doc=SimpleDocTemplate(path,pagesize=A4)
    styles=getSampleStyleSheet()
    story=[]
    story.append(Paragraph("<b>BILAN RACHIDIEN 3D</b>",styles["Heading1"]))
    story.append(Spacer(1,1*cm))
    story.append(Paragraph(f"{patient}",styles["Normal"]))
    story.append(Spacer(1,1*cm))
    story.append(Paragraph(f"Flèche Lombaire: {results['fl']:.2f} cm",styles["Normal"]))
    story.append(Spacer(1,1*cm))
    story.append(PDFImage(img_f,6*cm,9*cm))
    story.append(PDFImage(img_s,6*cm,9*cm))
    doc.build(story)
    return path

# ==============================
# UI
# ==============================
with st.sidebar:
    nom=st.text_input("Nom","DURAND")
    prenom=st.text_input("Prénom","Jean")
    remove_shoulders=st.toggle("Supprimer épaules",True)
    strong_smooth=st.toggle("Lissage fort",True)
    ply_file=st.file_uploader("Charger Scan (.PLY)",type=["ply"])

st.title("🦴 SpineScan Pro")

if ply_file:
    if st.button("LANCER ANALYSE"):
        pts=load_ply_numpy(ply_file)*0.1

        # projection clinique 3D->2D
        proj,basis,center=project_to_clinical_plane(pts)
        pts2d=np.column_stack([proj[:,0],pts[:,1],proj[:,1]])

        spine=anatomical_midline(pts2d,remove_shoulders=remove_shoulders)

        if strong_smooth:
            spine=smooth_spine(spine,window=81)

        fd,fl,vertical=sagittal_metrics(spine)

        # graphes
        tmp=tempfile.gettempdir()
        img_f=os.path.join(tmp,"f.png")
        img_s=os.path.join(tmp,"s.png")

        fig_f,ax=plt.subplots(figsize=(2.5,4))
        ax.scatter(pts2d[:,0],pts2d[:,1],s=0.2,alpha=0.1)
        ax.plot(spine[:,0],spine[:,1],'r',linewidth=2)
        ax.set_title("Frontale clinique")
        ax.axis("off")
        fig_f.savefig(img_f,dpi=150)

        fig_s,ax2=plt.subplots(figsize=(2.5,4))
        ax2.scatter(pts2d[:,2],pts2d[:,1],s=0.2,alpha=0.1)
        ax2.plot(spine[:,2],spine[:,1],'b',linewidth=2)
        ax2.plot(vertical,spine[:,1],'k--')
        ax2.set_title("Sagittale clinique")
        ax2.axis("off")
        fig_s.savefig(img_s,dpi=150)

        st.pyplot(fig_f)
        st.pyplot(fig_s)

        st.markdown(f"### Flèche lombaire: {fl:.2f} cm")

        pdf=export_pdf(f"{prenom} {nom}",{"fl":fl},img_f,img_s)
        with open(pdf,"rb") as f:
            st.download_button("Télécharger PDF",f,"rapport.pdf")
else:
    st.info("Importer un scan .PLY")
