"""
Render front-view comparison images for large beta variations.
Shows original, -2.0, -1.0, +1.0, +2.0 for each beta.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_obj(filepath):
    """Load OBJ file and return vertices and faces."""
    vertices = []
    faces = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    return np.array(vertices), np.array(faces)


def render_mesh_front(mesh_path, title, ax):
    """Render mesh from front view on given axes."""
    verts, faces = load_obj(mesh_path)

    # Subsample faces for faster rendering
    face_indices = np.arange(0, len(faces), 8)
    faces_sub = faces[face_indices]

    # For front view: swap axes so Y is up, Z is depth
    # Original: X=right, Y=up, Z=forward in SKEL
    verts_plot = verts[:, [0, 2, 1]]  # X, Z, Y -> X, Y_plot, Z_plot (for display)

    polygons = verts_plot[faces_sub]

    mesh = Poly3DCollection(polygons, alpha=0.95)
    mesh.set_facecolor([0.7, 0.7, 0.9])
    mesh.set_edgecolor([0.3, 0.3, 0.5])
    mesh.set_linewidth(0.1)
    ax.add_collection3d(mesh)

    # Set limits
    center = verts_plot.mean(axis=0)
    max_range = np.abs(verts_plot - center).max() * 1.2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Front view
    ax.view_init(elev=0, azim=0)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


def main():
    MESH_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/beta_large_variations'
    OUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/beta_renders_large'
    os.makedirs(OUT_DIR, exist_ok=True)

    # Beta0 comparison: -2.0, -1.0, original, +1.0, +2.0
    print("Rendering Beta[0] comparison...")
    fig = plt.figure(figsize=(20, 4))

    meshes_beta0 = [
        ('beta0_minus2.0.obj', 'Beta[0] -2.0\nSW=388.8mm'),
        ('beta0_minus1.0.obj', 'Beta[0] -1.0\nSW=378.1mm'),
        ('original.obj', 'Original\nSW=367.3mm'),
        ('beta0_plus1.0.obj', 'Beta[0] +1.0\nSW=356.6mm'),
        ('beta0_plus2.0.obj', 'Beta[0] +2.0\nSW=345.9mm'),
    ]

    for i, (mesh_file, title) in enumerate(meshes_beta0):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        mesh_path = os.path.join(MESH_DIR, mesh_file)
        render_mesh_front(mesh_path, title, ax)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'beta0_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

    # Beta1 comparison: -2.0, -1.0, original, +1.0, +2.0
    print("Rendering Beta[1] comparison...")
    fig = plt.figure(figsize=(20, 4))

    meshes_beta1 = [
        ('beta1_minus2.0.obj', 'Beta[1] -2.0\nSW=391.6mm'),
        ('beta1_minus1.0.obj', 'Beta[1] -1.0\nSW=379.5mm'),
        ('original.obj', 'Original\nSW=367.3mm'),
        ('beta1_plus1.0.obj', 'Beta[1] +1.0\nSW=355.2mm'),
        ('beta1_plus2.0.obj', 'Beta[1] +2.0\nSW=343.1mm'),
    ]

    for i, (mesh_file, title) in enumerate(meshes_beta1):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        mesh_path = os.path.join(MESH_DIR, mesh_file)
        render_mesh_front(mesh_path, title, ax)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'beta1_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

    # Combined: show extreme variations side by side
    print("Rendering extreme comparison...")
    fig = plt.figure(figsize=(20, 8))

    meshes_extreme = [
        # Top row: Beta[0]
        ('beta0_minus2.0.obj', 'Beta[0] -2.0\nSW=389mm'),
        ('original.obj', 'Original\nSW=367mm'),
        ('beta0_plus2.0.obj', 'Beta[0] +2.0\nSW=346mm'),
        # Bottom row: Beta[1]
        ('beta1_minus2.0.obj', 'Beta[1] -2.0\nSW=392mm'),
        ('original.obj', 'Original\nSW=367mm'),
        ('beta1_plus2.0.obj', 'Beta[1] +2.0\nSW=343mm'),
    ]

    for i, (mesh_file, title) in enumerate(meshes_extreme):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        mesh_path = os.path.join(MESH_DIR, mesh_file)
        render_mesh_front(mesh_path, title, ax)

    plt.suptitle('Beta Parameter Effects on Body Shape (±2.0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'extreme_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

    print(f"\nAll renders saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
