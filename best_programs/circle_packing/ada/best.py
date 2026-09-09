# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Explicit 26-circle constructor in a unit square using a staggered hex pattern with two vertical pitches.
    Design:
      - Use rows [5,4,5,4,5,3] with a single horizontal pitch a and two vertical pitches: b for the first 5 gaps except the last,
        and bt for the final gap (row 4 to row 5).
      - Maximize sum of radii by:
         1) Filling width with the 5-count rows in horizontal tangency: set a = 2*r5 and 2*r5 + 4*a = 1 => r5 = 0.1, a = 0.2.
         2) Saturating horizontal tangency in 4-count rows: set r4 = a/2 = 0.1.
         3) Enforcing diagonal tangency between 5- and 4-count rows: sqrt((a/2)^2 + b^2) = r5 + r4 = 0.2
            => b = sqrt(0.2^2 - 0.1^2) = sqrt(0.03).
         4) Let top gap bt and top radius r3 satisfy: vertical span r5 + 4*b + bt + r3 = 1 and diagonal tangency
            sqrt((a/2)^2 + bt^2) = r5 + r3. Solving gives:
               c = 1 - 4*b,
               bt = (c^2 - (a/2)^2) / (2c),
               r3 = c - bt - r5 = ((c - 0.1)^2) / (2c).
      - Odd rows are offset by a/2. Final radii are shrunk by tiny eps to ensure strict feasibility.
    Returns (centers(26,2), radii(26,), sum of radii).
    """
    r5 = 0.1
    a = 2.0 * r5  # 0.2
    b = np.sqrt(0.04 - (a/2.0)**2)  # sqrt(0.03)
    r4 = a / 2.0  # 0.1
    c = 1.0 - 4.0 * b
    bt = ((c*c) - (a/2.0)**2) / (2.0 * c)
    r3 = c - bt - r5

    rc = [5, 4, 5, 4, 5, 3]
    rr = [r5, r4, r5, r4, r5, r3]
    C, R = [], []
    for j, cnt in enumerate(rc):
        y = r5 + (j * b if j < 5 else 4.0 * b + bt)
        x0 = r5 + (a / 2.0 if j % 2 else 0.0)
        for k in range(cnt):
            C.append([x0 + k * a, y])
            R.append(rr[j])

    centers = np.array(C, float)
    radii = np.array(R, float) - 1e-12  # tiny shrink for strict inequalities
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
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)