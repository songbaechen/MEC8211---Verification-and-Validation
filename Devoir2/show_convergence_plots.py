"""
Affiche à la fin les figures de convergence déjà sauvegardées.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main() -> None:
    """
    Charge et affiche les figures de convergence espace et temps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

    img_space = mpimg.imread("convergence_space.png")
    img_time = mpimg.imread("convergence_time.png")

    axes[0].imshow(img_space)
    axes[0].axis("off")
    # axes[0].set_title("Convergence en espace")

    axes[1].imshow(img_time)
    axes[1].axis("off")
    # axes[1].set_title("Convergence en temps")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()