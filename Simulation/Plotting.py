# Plotting tools
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from astropy.coordinates import Angle
import astropy.units as u

def wrapped_rad2deg(lat_r, lon_r):
    """
    Converts latitude/longitude from radians to degrees.
    Longitude output is wrapped to [-180, 180) degrees.
    """
    lat_d = Angle(lat_r, unit=u.rad).to_value(u.deg)
    lon_d = Angle(lon_r, unit=u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d

def cart2eq(x, y, z):
    """
    Convert cartesian coordinates to spherical (r, lat, lon).
    lat: latitude in radians [-pi/2, pi/2]
    lon: longitude in radians [-pi, pi]
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return r, lat, lon

def eq2cart(r, lat, lon):
    """
    Convert spherical (r, lat, lon) to cartesian coordinates.
    """
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.vstack((x, y, z))


def draw_spherical_mesh(I, R, cmap='Inferno', bgcolor='rgb(70,70,70)'):
    """
    Interactive 3D spherical sky map using intensity values and visual caps.

    Parameters
    ----------
    I : (N, N_px) ndarray
        Intensity values per direction.
    R : (3, N_px) ndarray
        Cartesian coordinates (x, y, z) of pixels
    cmap : str
        Plotly color scale name (e.g. 'Inferno', 'Viridis', etc.).
    bgcolor : str
        Plotly background color for the scene.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive mesh figure.
    """

    # Convert full R to (lat, lon) and filter longitude
    _, lat, lon = cart2eq(R[0], R[1], R[2])
    _, lon_deg = wrapped_rad2deg(lat, lon)
    mask = (lon_deg >= -180) & (lon_deg <= 180)
    R_equatorial = eq2cart(1, lat[mask], lon[mask])

    x, y, z = R_equatorial

    # Convert to spherical coordinates
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    lat_deg = 90 - np.rad2deg(theta)

    # Normalize intensity
    I_norm = np.clip((I - I.min()) / (I.ptp() + 1e-12), 0, 1)

    # Triangulation
    triang = mtri.Triangulation(phi, theta)

    fig = go.Figure()

    # Main spherical mesh
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        i=triang.triangles[:, 0],
        j=triang.triangles[:, 1],
        k=triang.triangles[:, 2],
        intensity=I_norm,
        colorscale=cmap,
        showscale=True,
        opacity=1.0,
        lighting=dict(ambient=0.7, diffuse=1.0, specular=0.05, roughness=0.9),
        lightposition=dict(x=100, y=100, z=100),
        name='Sky Intensity',
    ))

    # Add visual caps
    def add_cap(z_cap, normal_up=True):
        """Add circular cap at z=z_cap."""
        n_segments = 100
        angles = np.linspace(0, 2 * np.pi, n_segments)
        radius = np.sqrt(1 - z_cap**2)
        cx = radius * np.cos(angles)
        cy = radius * np.sin(angles)
        cz = np.full_like(cx, z_cap)

        # Add center point
        cx = np.append(cx, 0)
        cy = np.append(cy, 0)
        cz = np.append(cz, z_cap)
        center_index = len(cx) - 1

        # Create triangles forming the cap
        triangles = []
        for i in range(n_segments - 1):
            a, b = i, i + 1
            if normal_up:
                triangles.append((a, b, center_index))
            else:
                triangles.append((b, a, center_index))
        triangles.append((n_segments - 1, 0, center_index) if normal_up else (0, n_segments - 1, center_index))

        tri_i, tri_j, tri_k = zip(*triangles)

        fig.add_trace(go.Mesh3d(
            x=cx, y=cy, z=cz,
            i=tri_i, j=tri_j, k=tri_k,
            color=bgcolor,
            opacity=1.0,
            showscale=False,
            name='Cap',
            lighting=dict(ambient=1.0, diffuse=0.0),
        ))

    # Get z-limits from original R
    z_min = np.min(z)
    z_max = np.max(z)

    add_cap(z_min, normal_up=False)
    add_cap(z_max, normal_up=True)

    # Final layout
    fig.update_layout(
        title='3D Spherical Sky Map (Capped)',
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=-1.5, y=0, z=0)),
            bgcolor=bgcolor
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig

def plot_PSNR(psnr, title, save_path=None):
    N_frames = len(psnr)
    
    # Plot the PSNR values per frame, with x-axis starting at 0 and ending at the last frame
    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(0, N_frames), psnr, 'o-')
    plt.title(title)
    plt.xlabel('Frame number')
    plt.ylabel('PSNR (dB)')
    plt.xlim(0, N_frames - 1)
    plt.grid()
    plt.tight_layout()
    # If required, save the plot to a file
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()