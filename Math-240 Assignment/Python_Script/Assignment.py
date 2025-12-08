import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import null_space
import os

# --- CONFIGURATION ---
output_folder = "Linear_Algebra_Project_JPEG"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"--- Processing started. Output folder: /{output_folder} ---")

# 1. SETUP & HELPER FUNCTIONS

def get_rebel_shape():
    """Generates the 3x100 matrix for a 3D Helix."""
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 3
    return np.vstack([x, y, z])

def get_math_stats(matrix):
    """Calculates Linear Algebra properties for the matrix."""
    rank = np.linalg.matrix_rank(matrix)
    
    # Null Space (Kernel)
    ns = null_space(matrix)
    nullity = ns.shape[1] if ns.size > 0 else 0
    
    # For a 3x3 matrix mapping R3 to R3:
    is_onto = "Yes" if rank == 3 else "No"
    is_one_to_one = "Yes" if nullity == 0 else "No"
    bijective = "Yes" if (is_onto == "Yes" and is_one_to_one == "Yes") else "No"
    
    if nullity == 0:
        kernel_text = "{(0,0,0)} (Trivial)"
    else:
        kernel_text = f"Span of {nullity} vector(s)"

    return (
        f"Rank:       {rank}\n"
        f"Nullity:    {nullity}\n"
        f"Kernel:     {kernel_text}\n"
        f"----------------\n"
        f"One-to-One: {is_one_to_one}\n"
        f"Onto:       {is_onto}\n"
        f"Bijective:  {bijective}"
    )

def setup_figure_layout(title, matrix):
    """Creates a standardized figure with space for Plot (Left) and Stats (Right)."""
    fig = plt.figure(figsize=(14, 7))
    
    # 3D Plot Area (Left)
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(title, fontsize=14, fontweight='bold')
    
    # Math Stats Area (Right)
    ax_text = fig.add_subplot(1, 2, 2)
    ax_text.axis('off')
    
    stats_str = get_math_stats(matrix)
    display_text = (
        f"TRANSFORMATION:\n{title}\n\n"
        f"MATRIX ({matrix.shape[0]}x{matrix.shape[1]}):\n{np.round(matrix, 2)}\n\n"
        f"PROPERTIES:\n{stats_str}"
    )
    
    ax_text.text(0.1, 0.5, display_text, fontsize=13, family='monospace', verticalalignment='center')
    
    return fig, ax_3d

def save_static_jpg(original, transformed, matrix, title, filename):
    print(f"Generating JPEG: {filename}...")
    
    fig, ax = setup_figure_layout(title, matrix)
    
    # Plot Original (Gray dashed)
    ax.plot(original[0,:], original[1,:], original[2,:], 'k--', alpha=0.3, label='Original')
    
    # Plot Transformed (Blue solid)
    ax.plot(transformed[0,:], transformed[1,:], transformed[2,:], 'b-', linewidth=2, label='Result')
    ax.scatter(transformed[0,:], transformed[1,:], transformed[2,:], c=transformed[2,:], cmap='cool', s=20)
    
    ax.legend()
    
    save_path = os.path.join(output_folder, filename)
    # Save as JPG with 95% quality (high res)
    plt.savefig(save_path, dpi=150, format='jpg', bbox_inches='tight')
    plt.close(fig)

def save_animation(points, frame_generator, base_matrix, title, filename):
    print(f"Rendering Animation: {filename}...")
    
    fig, ax = setup_figure_layout(title, base_matrix)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 5)
    
    graph, = ax.plot([], [], [], 'b-', linewidth=2, label='Transformed')
    scatter = ax.scatter([], [], [], c=[], cmap='cool', s=20)
    ax.plot(points[0,:], points[1,:], points[2,:], 'k--', alpha=0.2, label='Original')
    ax.legend()

    def update(frame):
        new_points = frame_generator(frame, points)
        graph.set_data(new_points[0,:], new_points[1,:])
        graph.set_3d_properties(new_points[2,:])
        scatter._offsets3d = (new_points[0,:], new_points[1,:], new_points[2,:])
        return graph, scatter

    anim = FuncAnimation(fig, update, frames=np.arange(0, 40), interval=80, blit=False)
    
    save_path = os.path.join(output_folder, filename)
    anim.save(save_path, writer=PillowWriter(fps=15))
    plt.close(fig)


# 2.EXECUTION

points = get_rebel_shape()

# --- STEP 0: REFERENCE IMAGE (Identity) ---
Identity = np.eye(3)
save_static_jpg(points, points, Identity, "Reference (Identity)", "0_Reference_Original.jpg")

# --- A. ROTATION (Animated GIF) ---
def rotate_logic(frame, pts):
    angle = np.radians(frame * 9) 
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return R @ pts
theta = np.radians(30)
R_stat = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
save_animation(points, rotate_logic, R_stat, "Rotation (Animated)", "A_Rotation.gif")

# --- B. REFLECT X-AXIS ---
T_ref_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_x @ points, T_ref_x, "Reflect X-Axis", "B_Reflect_X.jpg")

# --- C. REFLECT Y-AXIS ---
T_ref_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_y @ points, T_ref_y, "Reflect Y-Axis", "C_Reflect_Y.jpg")

# --- D. REFLECT LINE y=x ---
T_ref_yx = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
save_static_jpg(points, T_ref_yx @ points, T_ref_yx, "Reflect Line y=x", "D_Reflect_YX.jpg")

# --- E. REFLECT LINE y=kx (k=2) ---
k = 2; factor = 1 / (1 + k**2)
T_ref_kx = factor * np.array([[1 - k**2, 2*k, 0], [2*k, k**2 - 1, 0], [0, 0, 1 + k**2]])
T_ref_kx[2,2] = 1 
save_static_jpg(points, T_ref_kx @ points, T_ref_kx, f"Reflect Line y={k}x", "E_Reflect_KX.jpg")

# --- F. SHRINK/ENLARGE (Animated GIF) ---
def scale_logic(frame, pts):
    s = 1 + 0.5 * np.sin(frame / 5)
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, s]]) @ pts
S_stat = np.eye(3) * 2 
save_animation(points, scale_logic, S_stat, "Shrink/Enlarge (Animated)", "F_Scaling.gif")

# --- G. SHEAR X (Animated GIF) ---
def shear_x_logic(frame, pts):
    sf = 2 * np.sin(frame / 10)
    return np.array([[1, sf, 0], [0, 1, 0], [0, 0, 1]]) @ pts
Sh_x_stat = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
save_animation(points, shear_x_logic, Sh_x_stat, "Shear X-Axis (Animated)", "G_Shear_X.gif")

# --- H. SHEAR Y (Animated GIF) ---
def shear_y_logic(frame, pts):
    sf = 2 * np.sin(frame / 10)
    return np.array([[1, 0, 0], [sf, 1, 0], [0, 0, 1]]) @ pts
Sh_y_stat = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
save_animation(points, shear_y_logic, Sh_y_stat, "Shear Y-Axis (Animated)", "H_Shear_Y.gif")

# --- I. COMBINATION ---
Combo = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ \
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ \
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
save_static_jpg(points, Combo @ points, Combo, "Combination (Shear->Rot->Ref)", "I_Combination.jpg")

# --- J. TRANSLATION (Bonus Case) ---
print("Generating JPEG: J_Translation.jpg...")
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(1, 2, 1, projection='3d')
dx, dy = 2, 3
pts_4d = np.vstack([points, np.ones((1, points.shape[1]))])
T_trans = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, 0], [0, 0, 0, 1]])
shifted = (T_trans @ pts_4d)[:3, :]

ax.plot(points[0,:], points[1,:], points[2,:], 'k--', alpha=0.3, label='Original')
ax.plot(shifted[0,:], shifted[1,:], shifted[2,:], 'r-', linewidth=2, label='Shifted')
ax.legend()
ax.set_title("Translation (x+2, y+3)")

ax_txt = fig.add_subplot(1, 2, 2); ax_txt.axis('off')
info = (
    "TRANSLATION IS NOT A LINEAR TRANSFORMATION\n"
    "(It shifts the origin, so T(0) != 0)\n\n"
    "Technique Used: Homogeneous Coordinates\n"
    f"Shift: x + {dx}, y + {dy}"
)
ax_txt.text(0.1, 0.5, info, fontsize=12, family='monospace')

plt.savefig(os.path.join(output_folder, "J_Translation.jpg"), dpi=150, format='jpg', bbox_inches='tight')
plt.close(fig)

print("\n--- All Done! Check the folder 'Linear_Algebra_Project_JPEG' ---")
