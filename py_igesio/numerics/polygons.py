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
        on_curveがTrueの頂点については、昇順で並んでいることが期待される。
    """
    vertices: np.ndarray
    on_curve: list[bool]
    curve_params: list[float]

    def count(self) -> int:
        """頂点数 M"""
        return len(self.vertices)

    def get_curve_param_index(self, edge_index: int) -> tuple[int, int]:
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
            曲線パラメータ範囲の境界となる頂点のインデックス (i_start, i_end)。
            i_start, i_end はいずれも on_curve=True の頂点を指す。
            パラメータ範囲をまたぐ場合 (曲線の末尾側～先頭側の場合) は、i_start > i_end となる。

        Raises
        ------
        ValueError
            edge_index が範囲外の場合。すべての頂点が on_curve=False である場合。
        """
        m = len(self.vertices)
        if not 0 <= edge_index < m:
            raise ValueError(f"edge_index must be in [0, {m - 1}], got {edge_index}.")
        all_false = all(not flag for flag in self.on_curve)
        if all_false:
            raise ValueError("All vertices have on_curve=False. "
                             "Cannot determine curve parameter range.")

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

        return start, end

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
        全ての頂点は曲線C(t)上にある (on_curve=True)
    """
    circumscribed: PolygonData
    inscribed: PolygonData
    approximate: PolygonData


def _winding_number(point: np.ndarray, vertices: np.ndarray) -> int:
    """Non-Zero Winding則に基づくwinding numberを計算する。

    Parameters
    ----------
    point : np.ndarray
        判定する点。shape は (2,)。
    vertices : np.ndarray
        多角形の頂点座標。shape は (N, 2)。

    Returns
    -------
    int
        Winding number。非ゼロなら点は多角形の内部に存在する。
    """
    winding = 0
    n = len(vertices)
    px, py = float(point[0]), float(point[1])
    for i in range(n):
        ax, ay = float(vertices[i, 0]), float(vertices[i, 1])
        bx, by = float(vertices[(i + 1) % n, 0]), float(vertices[(i + 1) % n, 1])
        if ay <= py:
            if by > py:
                # 上向き交差: 点が辺の左側なら +1
                if (bx - ax) * (py - ay) - (by - ay) * (px - ax) > 0:
                    winding += 1
        else:
            if by <= py:
                # 下向き交差: 点が辺の右側なら -1
                if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < 0:
                    winding -= 1
    return winding


def _point_segment_dist_sq(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点 p から線分 (a, b) への距離の二乗を返す。"""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq == 0.0:
        d = p - a
        return float(np.dot(d, d))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_sq))
    d = p - (a + t * ab)
    return float(np.dot(d, d))


def _find_nearest_edge(point: np.ndarray, polygon: PolygonData) -> tuple[int, float]:
    """多角形の全辺の中から、点に最も近い辺のインデックスと距離の二乗を返す。

    Parameters
    ----------
    point : np.ndarray
        判定する点。shape は (2,)。
    polygon : PolygonData
        対象の多角形。

    Returns
    -------
    tuple[int, float]
        (最近傍辺のインデックス, 距離の二乗)
    """
    m = polygon.count()
    min_d = float('inf')
    nearest = 0
    for i in range(m):
        d = _point_segment_dist_sq(
            point, polygon.vertices[i], polygon.vertices[(i + 1) % m])
        if d < min_d:
            min_d = d
            nearest = i
    return nearest, min_d


def _extract_vertices_range(
        polygon: PolygonData, i_start: int, i_end: int, is_wrap: bool) -> np.ndarray:
    """多角形から i_start から i_end (両端含む) の頂点列を返す。

    Parameters
    ----------
    polygon : PolygonData
        対象の多角形。
    i_start : int
        開始頂点インデックス。
    i_end : int
        終了頂点インデックス。
    is_wrap : bool
        True の場合は折り返し (末尾→先頭) 処理を行う。

    Returns
    -------
    np.ndarray
        頂点列。shape は (N, 2)。
    """
    verts = polygon.vertices
    if not is_wrap:
        return verts[i_start:i_end + 1]
    return np.vstack([verts[i_start:], verts[:i_end + 1]])


def _extract_approx_by_param(
        approx: PolygonData,
        param_start: float,
        param_end: float,
        is_wrap: bool) -> np.ndarray:
    """近似多角形から指定パラメータ範囲に対応する頂点列を返す。

    Parameters
    ----------
    approx : PolygonData
        近似多角形。全頂点は on_curve=True かつ curve_params は昇順。
    param_start : float
        パラメータ範囲の開始値。
    param_end : float
        パラメータ範囲の終了値。
    is_wrap : bool
        True の場合は折り返し (param_start > param_end) として処理する。

    Returns
    -------
    np.ndarray
        対応する頂点列。shape は (N, 2)。param_start から param_end の順に並ぶ。
    """
    params = np.array(approx.curve_params)
    if not is_wrap:
        mask = (params >= param_start) & (params <= param_end)
        return approx.vertices[mask]
    # 折り返し: param_start 以上の部分 (後半) → param_end 以下の部分 (前半) の順
    part1 = approx.vertices[params >= param_start]
    part2 = approx.vertices[params <= param_end]
    return np.vstack([part1, part2])


def is_point_in_polygon(point: np.ndarray, polygons: CurveContainmentPolygons) -> bool:
    """閉曲線 C(u) に対して、点が曲線内部に存在するかを判定する。

    CurveContainmentPolygons に格納された外包多角形・内包多角形・近似多角形を
    利用して内外判定を行う。内外判定は Non-Zero Winding 則を使用する。

    Parameters
    ----------
    point : np.ndarray
        判定する点。shape は (2,)。
    polygons : CurveContainmentPolygons
        閉曲線 C(u) の内包・外包・近似多角形。

    Returns
    -------
    bool
        点が曲線 C(u) の内部に存在する場合 True。
    """
    # Step 1: 外包多角形の外部 → 外部
    if _winding_number(point, polygons.circumscribed.vertices) == 0:
        return False

    # Step 2: 内包多角形の内部 → 内部
    if _winding_number(point, polygons.inscribed.vertices) != 0:
        return True

    # Step 3: 詳細判定
    # 3-1: 内包・外包多角形の全辺から最近傍辺を探索
    circ_idx, circ_d = _find_nearest_edge(point, polygons.circumscribed)
    insc_idx, insc_d = _find_nearest_edge(point, polygons.inscribed)

    if insc_d <= circ_d:
        nearest_poly = polygons.inscribed
        nearest_edge = insc_idx
        is_inscribed = True
    else:
        nearest_poly = polygons.circumscribed
        nearest_edge = circ_idx
        is_inscribed = False

    # 3-2: パラメータ範囲の取得
    # get_curve_param_index は両端が on_curve=True の境界頂点インデックス (i_start, i_end) を返す。
    # 折り返し判定: i_start > i_end の場合に折り返しとする。
    i_start, i_end = nearest_poly.get_curve_param_index(nearest_edge)
    is_wrap = i_start > i_end

    poly_verts = _extract_vertices_range(nearest_poly, i_start, i_end, is_wrap)
    param_start = nearest_poly.curve_params[i_start]
    param_end = nearest_poly.curve_params[i_end]

    # 3-3: 近似多角形から対応するパラメータ範囲の頂点を取得
    approx_verts = _extract_approx_by_param(
        polygons.approximate, param_start, param_end, is_wrap)

    # 3-4: 再構成多角形の構築
    # poly_verts[0] ≈ approx_verts[0] (param_start 上)
    # poly_verts[-1] ≈ approx_verts[-1] (param_end 上)
    # 閉多角形: poly_verts (順方向) + approx_verts (逆方向) で共有端点を重複させない
    reconstructed = np.vstack([poly_verts[:-1], approx_verts[::-1][:-1]])

    # 3-5: 再構成多角形による内外判定
    inside = _winding_number(point, reconstructed) != 0
    # 内包多角形の辺を使用: 内部→内部、外部→外部
    # 外包多角形の辺を使用: 内部→外部（反転）、外部→内部（反転）
    return inside if is_inscribed else not inside
