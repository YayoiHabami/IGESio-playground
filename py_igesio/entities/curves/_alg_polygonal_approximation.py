"""
曲線の離散化アルゴリズム

NOTE: C++移植時は`src/entities/curves/algorithms/polygonal_approximation.h(cpp)`に実装する (APIに含めない) 。
"""
import numpy as np

from py_igesio.numerics.polygons import PolygonData
from ..interfaces import CurveBase2D

def _is_in_linear_segment(
    u_a: float,
    u_b: float,
    linear_segments: list[tuple[float, float]],
) -> bool:
    """区間 [u_a, u_b] が直線部に完全に含まれるかを判定する。"""
    for u_s, u_e in linear_segments:
        if u_s <= u_a and u_b <= u_e:
            return True
    return False


def _point_to_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点 p から直線 a-b への距離（点-直線距離）を返す。

    a == b の場合は点 a との距離を返す。
    """
    ab = b - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-14:
        return float(np.linalg.norm(p - a))
    # |cross(ab, ap)| / |ab|
    ap = p - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    return abs(cross) / ab_norm


def _subdivide(
    curve: CurveBase2D,
    u_a: float,
    u_b: float,
    eps: float,
    max_depth: int,
) -> list[float]:
    """反復スタックを用いた再帰的二分法で区間 [u_a, u_b] を折れ線近似する。

    Returns
    -------
    list[float]
        u_a から u_b までのパラメータ列（両端を含む）。
    """
    # スタックの各エントリ: (u_left, u_right, depth)
    # 「確定済みパラメータ」をリストで管理し、分割が不要になった区間から順に追加する。
    # 実装方針:
    #   - スタックには「まだ判定していない区間」を積む
    #   - 分割不要 → 左端を確定リストに追加（右端は次の区間の左端と重複するため後回し）
    #   - 分割必要 → 左・右サブ区間をスタックへ（左を後に積むことで深さ優先・左→右順を維持）

    # 結果を順序付きで管理するために、確定した区間端点を収集する
    # 深さ優先で左から処理するため、スタックに (u_a, u_b, depth) を積む際は
    # 右サブ区間 → 左サブ区間の順で積む（後入れ先出しで左が先に処理される）

    confirmed: list[float] = [u_a]
    stack: list[tuple[float, float, int]] = [(u_a, u_b, 0)]
    linear_segments = curve.get_linear_segments()

    while stack:
        ua, ub, depth = stack.pop()

        # --- 終了条件 1: 直線部 ---
        if _is_in_linear_segment(ua, ub, linear_segments):
            confirmed.append(ub)
            continue

        # --- 終了条件 2: 中点距離が許容値以下 ---
        u_mid = (ua + ub) / 2.0
        pa = curve.evaluate(ua)
        pb = curve.evaluate(ub)
        pm = curve.evaluate(u_mid)
        dist = _point_to_line_distance(pm, pa, pb)

        if dist <= eps:
            confirmed.append(ub)
            continue

        # --- 終了条件 3: 最大深度超過 ---
        # ua はすでに confirmed に存在するため u_mid と ub のみ追記する
        if depth >= max_depth:
            confirmed.append(u_mid)
            confirmed.append(ub)
            continue

        # --- 分割: 右サブ区間を先に積む（スタックはLIFOなので左が先に処理される）---
        stack.append((u_mid, ub, depth + 1))
        stack.append((ua, u_mid, depth + 1))

    return confirmed


#---------------------------------------------------------------------------
# 近似多角形の構築
#---------------------------------------------------------------------------

def compute_approximate_polygon(
    curve: CurveBase2D,
    inscribed: PolygonData,
    circumscribed: PolygonData,
    eps: float,
    max_depth: int = 1000,
) -> PolygonData:
    """曲線を折れ線近似した近似多角形を生成する。

    内接多角形・外接多角形の全頂点（on_curve=True のもの）を固定点として含み、
    各固定点間の弧を「辺と弧の中点との距離 <= eps」を満たすまで再帰的に二分する。

    Parameters
    ----------
    curve : CurveBase2D
        閉曲線。
    inscribed : PolygonData
        内接多角形（compute_inscribed_polygon の戻り値）。
    circumscribed : PolygonData
        外接多角形（compute_circumscribed_polygon の戻り値）。
    eps : float
        折れ線近似の許容距離（辺と弧の中点との点-直線距離）。
    max_depth : int
        反復スタックの最大深度。デフォルトは 1000。

    Returns
    -------
    PolygonData
        近似多角形。全頂点は曲線上（on_curve=True）。
    """
    u_min, u_max = curve.get_range()

    # ------------------------------------------------------------------
    # Step 1: 固定点パラメータの収集・ソート・重複除去
    # ------------------------------------------------------------------
    fixed_set: set[float] = set()
    for pd in (inscribed, circumscribed):
        for u, on in zip(pd.curve_params, pd.on_curve):
            if on:
                fixed_set.add(u)

    # u_min / u_max は同一点（閉曲線）なので両方を固定点として登録し、
    # 区間の番兵として機能させる
    fixed_set.add(u_min)
    fixed_set.add(u_max)

    fixed_params = sorted(fixed_set)

    # ------------------------------------------------------------------
    # Step 2: 各固定点間の区間を折れ線近似
    # ------------------------------------------------------------------
    result_params: list[float] = []

    for idx in range(len(fixed_params) - 1):
        u_a = fixed_params[idx]
        u_b = fixed_params[idx + 1]

        # 区間内の全パラメータを反復スタックで求める
        interval_params = _subdivide(
            curve=curve,
            u_a=u_a,
            u_b=u_b,
            eps=eps,
            max_depth=max_depth,
        )

        if idx == 0:
            result_params.extend(interval_params)
        else:
            # u_a は前区間の末尾と重複するため除く
            result_params.extend(interval_params[1:])

    # u_min == u_max（閉曲線の折り返し点）なので末尾の u_max を除く
    if len(result_params) > 1 and np.isclose(result_params[-1], u_max):
        result_params = result_params[:-1]

    # ------------------------------------------------------------------
    # Step 3: PolygonData の構築
    # ------------------------------------------------------------------
    vertices = np.array([curve.evaluate(u) for u in result_params])
    on_curve = [True] * len(result_params)
    curve_params = list(result_params)

    return PolygonData(
        vertices=vertices,
        on_curve=on_curve,
        curve_params=curve_params,
    )
