import numpy as np
from scipy.signal import savgol_filter

# ------------------------------
# Rotation helpers (tu peux garder les tiens)
# ------------------------------
def estimate_rotation_xz(pts):
    y = pts[:, 1]
    mid = (y > np.percentile(y, 30)) & (y < np.percentile(y, 70))
    pts_mid = pts[mid] if np.count_nonzero(mid) > 200 else pts
    XZ = pts_mid[:, [0, 2]]
    XZ = XZ - np.mean(XZ, axis=0)
    _, _, Vt = np.linalg.svd(XZ, full_matrices=False)
    angle = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
    c, s = np.cos(-angle), np.sin(-angle)
    return np.array([[c, -s], [s, c]], dtype=float)

def apply_rotation_xz(pts, R):
    XZ_rot = pts[:, [0, 2]] @ R.T
    return np.column_stack([XZ_rot[:, 0], pts[:, 1], XZ_rot[:, 1]])

# ------------------------------
# Density-free binning in X
# ------------------------------
def x_bins_present(sl, cell_cm=0.5, min_pts_per_bin=3):
    """
    Retourne les centres de bins X non vides.
    Chaque bin = 1 vote, donc pas de biais densité.
    """
    x = sl[:, 0]
    xmin, xmax = np.percentile(x, [2, 98])
    if xmax - xmin < 1e-6:
        return None

    nbins = max(30, int(np.ceil((xmax - xmin) / cell_cm)))
    edges = np.linspace(xmin, xmax, nbins + 1)

    centers = []
    for b in range(nbins):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if np.count_nonzero(m) >= min_pts_per_bin:
            centers.append(0.5 * (edges[b] + edges[b + 1]))

    if len(centers) < 10:
        return None
    return np.array(centers, dtype=float)

def midline_from_bins(xc, q=10):
    """
    Milieu robuste entre bords (bins) :
    x0 = (Pq + P(100-q))/2
    """
    xl = float(np.percentile(xc, q))
    xr = float(np.percentile(xc, 100 - q))
    return 0.5 * (xl + xr)

# ------------------------------
# Apophyse proxy within central band
# ------------------------------
def pick_spinous_proxy(sl, x_mid, band_cm=3.0, z_quantile_for_surface=92):
    """
    Dans une bande centrale autour de x_mid :
    - on se limite aux points "surface dos" via un quantile Z
    - puis on prend le max Z (plus postérieur)
    """
    x = sl[:, 0]
    z = sl[:, 2]

    band = (x >= x_mid - band_cm) & (x <= x_mid + band_cm)
    slb = sl[band] if np.count_nonzero(band) >= 25 else sl

    # garde seulement la surface (évite points internes/bruit)
    zthr = np.percentile(slb[:, 2], z_quantile_for_surface)
    surf = slb[slb[:, 2] >= zthr]
    use = surf if surf.shape[0] >= 15 else slb

    i = int(np.argmax(use[:, 2]))
    x0 = float(use[i, 0])
    z0 = float(np.percentile(use[:, 2], 90))
    return x0, z0

# ------------------------------
# Main extraction: SPINOUS
# ------------------------------
def extract_spine_spinous(pts, remove_shoulders=True,
                          band_cm=3.0,
                          cell_cm=0.45,
                          y_low=10, y_high=92,
                          n_slices_min=140, n_slices_max=240):
    """
    Extraction apophyses :
    1) rotation XZ
    2) slices Y
    3) midline density-free par bins X (milieu entre bords)
    4) apophyse proxy = max Z dans bande centrale +/- band_cm
    """
    R = estimate_rotation_xz(pts)
    pr = apply_rotation_xz(pts, R)

    y = pr[:, 1]
    y0 = np.percentile(y, y_low)
    y1 = np.percentile(y, y_high)

    n_slices = int(np.clip((pr.shape[0] // 3000) + 160, n_slices_min, n_slices_max))
    edges_y = np.linspace(y0, y1, n_slices)

    spine = []
    prev_x = None

    for i in range(len(edges_y) - 1):
        sl = pr[(y >= edges_y[i]) & (y < edges_y[i + 1])]
        if sl.shape[0] < 40:
            continue

        # Option épaules : on enlève juste le très haut si besoin (mais soft)
        # Ici: on ne "coupe" pas en Z fort, sinon on casse la symétrie.
        # On préfère filtrer les extrêmes X si la tranche est très large (épaules/bras).
        if remove_shoulders:
            # enlève extrêmes latéraux si ça ressemble à bras/artefacts
            x = sl[:, 0]
            xl, xr = np.percentile(x, [2, 98])
            sl = sl[(x >= xl) & (x <= xr)]
            if sl.shape[0] < 30:
                continue

        # --- 1) midline density-free (bins X) ---
        xc = x_bins_present(sl, cell_cm=cell_cm, min_pts_per_bin=3)
        if xc is None:
            continue
        x_mid = midline_from_bins(xc, q=12)

        # --- 2) apophyse proxy dans bande centrale ---
        x0, z0 = pick_spinous_proxy(sl, x_mid, band_cm=band_cm, z_quantile_for_surface=92)
        y_mid = float(np.median(sl[:, 1]))  # pas de moyenne

        # continuité douce (mais pas trop tôt)
        if prev_x is not None and len(spine) > 12:
            if abs(x0 - prev_x) > 2.5:
                x0 = prev_x
        prev_x = x0

        spine.append([x0, y_mid, z0])

    if len(spine) == 0:
        return np.empty((0, 3), dtype=float)

    spine = np.array(spine, dtype=float)
    spine = spine[np.argsort(spine[:, 1])]

    # retour repère original
    XZ_back = spine[:, [0, 2]] @ R
    spine[:, 0] = XZ_back[:, 0]
    spine[:, 2] = XZ_back[:, 1]
    return spine
