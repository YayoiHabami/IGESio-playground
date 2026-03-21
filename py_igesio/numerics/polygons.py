from dataclasses import dataclass

import numpy as np



@dataclass
class PolygonData:
    """多角形の頂点と曲線パラメータを保持するデータクラス。
    Attributes
    ----------
    vertices : np.ndarray
        多角形の頂点座標。shape は (M, 2)。
    on_curve : list[bool]
        各頂点が曲線 C(u) 上にあるか否か。長さ M。
    curve_params : list[float]
        各頂点の曲線 C(u) におけるパラメータ u。長さ M。
        対応する on_curve が False の場合は 0。
    """
    vertices: np.ndarray
    on_curve: list[bool]
    curve_params: list[float]

    def count(self) -> int:
        """頂点数 M"""
        return len(self.vertices)

    def get_curve_param_index(self, edge_index: int) -> tuple[float, float]:
        """指定された辺が含まれる曲線のパラメータ範囲に囲まれた頂点の開始/終了インデックスを返す。

        辺のインデックス i は、頂点 i から頂点 (i+1) % M への辺を表す。
        両端点のうち on_curve が False の頂点は制御点とみなし、
        その頂点を挟む on_curve が True の頂点までパラメータ範囲を拡張する。

        Parameters
        ----------
        edge_index : int
            辺のインデックス。0 以上 M 未満。

        Returns
        -------
        tuple[int, int]
            頂点のインデックス範囲 (u_start, u_end)。

        Raises
        ------
        ValueError
            edge_index が範囲外の場合。
        """
        m = len(self.vertices)
        if not 0 <= edge_index < m:
            raise ValueError(f"edge_index must be in [0, {m - 1}], got {edge_index}.")

        i = edge_index
        j = (edge_index + 1) % m

        # 始点側: on_curve が True になるまで遡る
        start = i
        while not self.on_curve[start]:
            start = (start - 1) % m

        # 終点側: on_curve が True になるまで進む
        end = j
        while not self.on_curve[end]:
            end = (end + 1) % m

        return start, (end - 1) % m

@dataclass
class CurveContainmentPolygons:
    """閉曲線C(t)の内包・外包多角形と、近似多角形を保持するデータクラス
    (閉曲線C(t)に対する内外判定の高速化のために使用する)

    Attributes
    ----------
    circumscribed : PolygonData
        曲線C(t)を完全に内包する多角形の頂点データ
    inscribed : PolygonData
        曲線C(t)に完全に内包される多角形の頂点データ
    approximate : PolygonData
        曲線C(t)を折れ線近似した多角形の頂点データ
    """
    circumscribed: PolygonData
    inscribed: PolygonData
    approximate: PolygonData
