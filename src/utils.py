import os

import seaborn as sns
import matplotlib.pyplot as plt


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_scatterplot_and_save(df, x, y, file_path):
    plt.rcParams["font.family"] = "SF Compact Text"
    font_scale = 2
    sns.set(font_scale=font_scale, font="SF Compact Text")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Cost")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(file_path)
