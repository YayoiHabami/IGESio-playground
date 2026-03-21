"""
曲線エンティティ全般に関連した関数の実装 (離散化等)

NOTE: `as _module_name`でインポートしているものは、C++移植時ソースファイル内でインクルードするようにする
      (ヘッダーファイルに定義しない (APIに含めない)). 関数名の先頭にアンダースコアを付けているものも同様の扱いとする。
"""
from py_igesio.numerics.polygons import CurveContainmentPolygons
from py_igesio.entities.interfaces import CurveBase2D
from ._alg_extremal_polygon import (
    compute_circumscribed_polygon,
    compute_inscribed_polygon
)
from ._alg_polygonal_approximation import compute_approximate_polygon

def compute_containment_polygons(
    curve: CurveBase2D,
    n_vert: int,
    curvature_eps: float = 1e-3,
    distance_eps: float = 1e-3
) -> CurveContainmentPolygons:
    """閉曲線curveに対して、内包・外包多角形と近似多角形を計算する。

    Parameters
    ----------
    curve : CurveBase2D
        対象の閉曲線エンティティ
    n_vert : int
        内包・外包多角形の頂点数（近似多角形はこの値を超えることがある）
    curvature_eps : float, optional
        曲率判定の閾値 (曲率の絶対値がこの値以下の区間は直線部とみなす)
    distance_eps : float, optional
        距離判定の閾値 (近似多角形の辺の二分点が曲線からこの距離以上離れている場合は、さらに分割して近似精度を上げる)

    Returns
    -------
    CurveContainmentPolygons
         内包・外包多角形と近似多角形を保持するデータクラス
     """
    # 内包・外包多角形の計算
    circumscribed = compute_circumscribed_polygon(curve, n_vert, curvature_eps)
    inscribed = compute_inscribed_polygon(curve, n_vert, curvature_eps)

    # 内包・外包多角形の頂点を含むように、近似多角形を計算
    approximate = compute_approximate_polygon(
        curve, inscribed, circumscribed, distance_eps
    )
    return CurveContainmentPolygons(
        circumscribed=circumscribed,
        inscribed=inscribed,
        approximate=approximate,
    )
