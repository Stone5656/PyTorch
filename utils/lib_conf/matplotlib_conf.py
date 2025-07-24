import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib

def setup_matplotlib_style(font_size: int = 12, figsize: tuple = (8, 6)) -> None:
    """
    matplotlibの描画設定と日本語フォントを有効化する。
    他のスクリプトからもimportして使えるように関数化。

    Parameters:
    ----------
    font_size : int
        全体の基本フォントサイズ（デフォルト: 12）
    figsize : tuple
        デフォルトの図のサイズ (width, height)（デフォルト: (8, 6)）
    """
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["figure.figsize"] = figsize
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.titlesize"] = font_size + 2
    mpl.rcParams["axes.labelsize"] = font_size
    mpl.rcParams["legend.fontsize"] = font_size - 2
    mpl.rcParams["xtick.labelsize"] = font_size - 2
    mpl.rcParams["ytick.labelsize"] = font_size - 2
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.color"] = "gray"
