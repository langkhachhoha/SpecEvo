# EVOLVE-BLOCK-START
"""Deterministic n=26 packing:
- 25 circles on a 5x5 grid with r=0.1 at centers {0.1,0.3,0.5,0.7,0.9}^2.
- 1 interstitial circle at (0.2,0.2) tangent to four neighbors, radius r1=r*(sqrt(2)-1).
This direct constructor is short, fast, and yields sum_radii ≈ 2.541421356.
"""

import math
import numpy as np

def construct_packing():
    """Return (centers, radii, sum_radii) for the 25-grid + one interstitial layout."""
    r = 0.1
    g = [r, 0.3, 0.5, 0.7, 1 - r]
    centers = [(x, y) for y in g for x in g]
    radii = [r] * 25
    centers.append((2*r, 2*r))            # interstitial in a grid cell center
    radii.append(r * (math.sqrt(2) - 1))  # tangent to 4 neighbors
    centers = np.array(centers, float)
    radii = np.array(radii, float)
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
    print(f"Sum of radii: {sum_radii}")
    # Reference: AlphaEvolve reported ~2.635 for n=26
    # Uncomment to visualize:
    # visualize(centers, radii)