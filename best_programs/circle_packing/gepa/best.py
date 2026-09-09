# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Explicit, closed-form 26-circle constructor:
    - Place 25 equal circles of radius r=0.1 at the 5x5 grid {0.1,0.3,0.5,0.7,0.9}^2.
    - Add one extra circle at the center of any grid cell, e.g., (0.2,0.2).
      Its limiting neighbors are the four corners of that cell at distance sqrt(0.1^2+0.1^2),
      so its maximal radius is r_extra = sqrt(0.02) - 0.1.
    This arrangement is tangent-tight and non-iterative, yielding sum_radii ≈ 2.541421356.
    Returns:
      centers: (26,2) array, radii: (26,), sum_radii: float
    """
    # 25-grid centers
    g = 0.1 + 0.2*np.arange(5, dtype=float)
    gx, gy = np.meshgrid(g, g)
    centers = np.column_stack((gx.ravel(), gy.ravel()))

    # Radii: 25 at 0.1, plus one extra at a cell midpoint
    r_main = 0.1
    p_extra = np.array([0.2, 0.2], dtype=float)
    r_extra = np.sqrt(0.02) - 0.1

    centers = np.vstack([centers, p_extra])
    radii = np.concatenate([np.full(25, r_main, float), [r_extra]])
    return centers, radii, float(radii.sum())
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)