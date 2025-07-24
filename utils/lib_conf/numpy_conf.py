import numpy as np

def configure_numpy_print(
    precision: int = 4,
    suppress: bool = True,
    linewidth: int = 120,
    threshold: int = 1000,
    edgeitems: int = 3,
    sign: str = "-",  # or "+"
    floatmode: str = "maxprec"
) -> None:
    """
    NumPyの出力設定をカスタマイズして見やすくする。

    Parameters:
    -----------
    precision : int
        小数点以下の桁数
    suppress : bool
        指数表記の抑制（Trueで通常表記）
    linewidth : int
        出力時の1行の文字数上限
    threshold : int
        表示する最大要素数（超えると省略）
    edgeitems : int
        両端に表示する要素数
    sign : str
        符号の表示方法（"+", "-", " "）
    floatmode : str
        浮動小数の表示モード（"maxprec", "fixed", "unique"など）
    """
    np.set_printoptions(
        precision=precision,
        suppress=suppress,
        linewidth=linewidth,
        threshold=threshold,
        edgeitems=edgeitems,
        sign=sign,
        floatmode=floatmode
    )
