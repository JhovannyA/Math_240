import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import os

# --- CONFIGURATION ---
output_folder = "Linear_Algebra_Project_Canvas_Ready"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"--- Processing started. Output folder: /{output_folder} ---")

#HELPER FUNCTIONS

def get_rebel_shape():
    """Generates the 3x100 matrix for a 3D Helix."""
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 3
    return np.vstack([x, y, z])

def get_math_stats(matrix):
    """Calculates Linear Algebra properties."""
    rank = np.linalg.matrix_rank(matrix)
    ns = null_space(matrix)
    nullity = ns.shape[1] if ns.size > 0 else 0
    
    is_onto = "Yes" if rank == 3 else "No"
    is_one_to_one = "Yes" if nullity == 0 else "No"
    bijective = "Yes" if (is_onto == "Yes" and is_one_to_one == "Yes") else "No"
    
    if nullity == 0:
        kernel_text = "{(0,0,0)}"
    else:
        kernel_text = f"Span of {nullity} vec(s)"

    return (
        f"Rank: {rank} | Nullity: {nullity}\n"
        f"Kernel: {kernel_text}\n"
        f"1-to-1: {is_one_to_one} | Onto: {is_onto}\n"
        f"Bijective: {bijective}"
    )

def setup_single_plot(title, matrix):
    """Layout for single static images (Reflections)."""
    fig = plt.figure(figsize=(14, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_text = fig.add_subplot(1, 2, 2); ax_text.axis('off')
    
    stats_str = get_math_stats(matrix)
    display_text = f"TRANSFORMATION: {title}\n\nMATRIX:\n{np.round(matrix, 2)}\n\n{stats_str}"
    ax_text.text(0.1, 0.5, display_text, fontsize=12, family='monospace', verticalalignment='center')
    
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
    ax_3d.set_title(title, fontweight='bold')
    return fig, ax_3d

def save_static_jpg(original, transformed, matrix, title, filename):
    print(f"Generating Static JPEG: {filename}...")
    fig, ax = setup_single_plot(title, matrix)
    
    ax.plot(original[0,:], original[1,:], original[2,:], 'k--', alpha=0.3, label='Original')
    ax.plot(transformed[0,:], transformed[1,:], transformed[2,:], 'b-', linewidth=2, label='Result')
    ax.scatter(transformed[0,:], transformed[1,:], transformed[2,:], c=transformed[2,:], cmap='cool', s=20)
    ax.legend()
    
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=150, format='jpg', bbox_inches='tight')
    plt.close(fig)

def save_filmstrip(points, transform_func, base_matrix, title, filename):
    """Creates a 3-panel filmstrip to simulate animation in a static image."""
    print(f"Generating Filmstrip: {filename}...")
    
    fig = plt.figure(figsize=(18, 6))
    
    frames = [0, 20, 40]
    subtitles = ["Start", "Middle", "End"]
    
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(1, 4, i+1, projection='3d') # 1 row, 4 cols (3 plots + 1 text)
        new_points = transform_func(frame, points)
        
        ax.plot(points[0,:], points[1,:], points[2,:], 'k--', alpha=0.2) # Ghost original
        ax.plot(new_points[0,:], new_points[1,:], new_points[2,:], 'b-', linewidth=2)
        ax.scatter(new_points[0,:], new_points[1,:], new_points[2,:], c=new_points[2,:], cmap='cool', s=10)
        
        ax.set_title(f"{subtitles[i]} (t={frame})")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 5)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    ax_text = fig.add_subplot(1, 4, 4)
    ax_text.axis('off')
    stats_str = get_math_stats(base_matrix)
    display_text = f"ANIMATION SEQUENCE:\n{title}\n\nPROPERTIES:\n{stats_str}\n\n(Showing 3 keyframes)"
    ax_text.text(0.0, 0.5, display_text, fontsize=11, family='monospace', verticalalignment='center')

    plt.tight_layout()
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=150, format='jpg', bbox_inches='tight')
    plt.close(fig)

# 2. MAIN EXECUTION

points = get_rebel_shape()

#REFERENCE
Identity = np.eye(3)
save_static_jpg(points, points, Identity, "Reference (Identity)", "0_Reference.jpg")

#A. ROTATION (Filmstrip)
def rotate_logic(frame, pts):
    angle = np.radians(frame * 4) # Slower angle for filmstrip clarity
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return R @ pts
# Stats base
R_stat = np.array([[np.cos(0.5), -np.sin(0.5), 0], [np.sin(0.5), np.cos(0.5), 0], [0, 0, 1]])
save_filmstrip(points, rotate_logic, R_stat, "Rotation 30 deg", "A_Rotation_Filmstrip.jpg")

#B. REFLECT X
T_ref_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_x @ points, T_ref_x, "Reflect X-Axis", "B_Reflect_X.jpg")

#C. REFLECT Y
T_ref_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_y @ points, T_ref_y, "Reflect Y-Axis", "C_Reflect_Y.jpg")

#D. REFLECT y=x
T_ref_yx = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_yx @ points, T_ref_yx, "Reflect Line y=x", "D_Reflect_YX.jpg")

#E. REFLECT y=kx
k = 2; factor = 1 / (1 + k**2)
T_ref_kx = factor * np.array([[1 - k**2, 2*k, 0], [2*k, k**2 - 1, 0], [0, 0, 1 + k**2]])
T_ref_kx[2,2] = 1 
save_static_jpg(points, T_ref_kx @ points, T_ref_kx, f"Reflect Line y={k}x", "E_Reflect_KX.jpg")

#F. SHRINK/ENLARGE (Filmstrip)
def scale_logic(frame, pts):
    # Scale from 1.0 up to 2.0
    s = 1 + (frame / 40.0) 
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, s]]) @ pts
S_stat = np.eye(3) * 2 
save_filmstrip(points, scale_logic, S_stat, "Shrink/Enlarge", "F_Scaling_Filmstrip.jpg")

# --- G. SHEAR X (Filmstrip) ---
def shear_x_logic(frame, pts):
    sf = (frame / 20.0) # Shear factor increases
    return np.array([[1, sf, 0], [0, 1, 0], [0, 0, 1]]) @ pts
Sh_x_stat = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
save_filmstrip(points, shear_x_logic, Sh_x_stat, "Shear X-Axis", "G_Shear_X_Filmstrip.jpg")

#H. SHEAR Y (Filmstrip)
def shear_y_logic(frame, pts):
    sf = (frame / 20.0)
    return np.array([[1, 0, 0], [sf, 1, 0], [0, 0, 1]]) @ pts
Sh_y_stat = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
save_filmstrip(points, shear_y_logic, Sh_y_stat, "Shear Y-Axis", "H_Shear_Y_Filmstrip.jpg")

#I. COMBINATION
Combo = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ \
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ \
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
save_static_jpg(points, Combo @ points, Combo, "Combination (Shear->Rot->Ref)", "I_Combination.jpg")

#J. TRANSLATION
print("Generating Static JPEG: J_Translation.jpg...")
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
dx, dy = 2, 3
pts_4d = np.vstack([points, np.ones((1, points.shape[1]))])
T_trans = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, 0], [0, 0, 0, 1]])
shifted = (T_trans @ pts_4d)[:3, :]
ax.plot(points[0,:], points[1,:], points[2,:], 'k--', alpha=0.3, label='Original')
ax.plot(shifted[0,:], shifted[1,:], shifted[2,:], 'r-', linewidth=2, label='Shifted')
ax.legend(); ax.set_title("Translation (x+2, y+3)")
# Text
ax_txt = fig.add_subplot(1, 2, 2); ax_txt.axis('off')
ax_txt.text(0.1, 0.5, "TRANSLATION (Affine)\nShift Origin: x+2, y+3", fontsize=12, family='monospace')
plt.savefig(os.path.join(output_folder, "J_Translation.jpg"), dpi=150, format='jpg', bbox_inches='tight')
plt.close(fig)

print("\n--- All Done! Check folder: 'Linear_Algebra_Project_Canvas_Ready' ---")