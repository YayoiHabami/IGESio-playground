"""polygons.py の単体テスト"""
import math

import pytest
import numpy as np

from py_igesio.numerics.polygons import (
    PolygonData,
    CurveContainmentPolygons,
    _winding_number,
    _point_segment_dist_sq,
    _find_nearest_edge,
    _extract_vertices_range,
    _extract_approx_by_param,
    is_point_in_polygon,
)


# ---------------------------------------------------------------------------
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _square_polygon(r: float = 1.0) -> PolygonData:
    """半径 r の正方形 (CCW)。全頂点 on_curve=True。"""
    verts = np.array([[ r, -r], [ r,  r], [-r,  r], [-r, -r]], dtype=float)
    params = [0.0, 0.25, 0.5, 0.75]
    return PolygonData(verts, [True, True, True, True], params)


def _make_circle_polygons(n_approx: int = 16) -> CurveContainmentPolygons:
    """単位円の CurveContainmentPolygons を手動で構築する。

    - circumscribed: 単位円に外接する正方形。
        頂点は接点（on_curve=True, 曲線上）と角頂点（on_curve=False）が交互に並ぶ。
        接点は (1,0),(0,1),(-1,0),(0,-1) の 4 点で、いずれも単位円上かつ
        近似多角形の頂点と一致する。
    - inscribed: 単位円に内接する正方形（45° 回転）。
        全頂点が on_curve=True で単位円上に存在する。
        近似多角形の対応する param の頂点と数値的に一致する。
    - approximate: n_approx 頂点の正多角形（単位円上）。

    注意: inscribed の on_curve 頂点は曲線上（単位円上）に置く必要がある。
    アルゴリズムは「内包/外包多角形の on_curve 頂点は近似多角形に含まれる」
    ことを前提に端点を接合するためである。
    """
    # --- circumscribed: 接点(on, 曲線上) と角(off) が交互、合計 8 頂点 ---
    # 接点: (1,0),(0,1),(-1,0),(0,-1) は単位円上 (param = 0, 0.25, 0.5, 0.75)
    circ_verts = np.array([
        [ 1.0,  0.0],  # 0: on, param=0.00  (単位円上)
        [ 1.0,  1.0],  # 1: off
        [ 0.0,  1.0],  # 2: on, param=0.25  (単位円上)
        [-1.0,  1.0],  # 3: off
        [-1.0,  0.0],  # 4: on, param=0.50  (単位円上)
        [-1.0, -1.0],  # 5: off
        [ 0.0, -1.0],  # 6: on, param=0.75  (単位円上)
        [ 1.0, -1.0],  # 7: off
    ], dtype=float)
    circ_on     = [True, False, True, False, True, False, True, False]
    circ_params = [0.0,  0.0,   0.25, 0.0,  0.50, 0.0,  0.75, 0.0]
    circumscribed = PolygonData(circ_verts, circ_on, circ_params)

    # --- inscribed: 単位円に内接する正方形（45° 回転）---
    # 頂点を単位円上（param = 0.125, 0.375, 0.625, 0.875）に置く。
    # n_approx=16 の近似多角形においてこれらの param と完全に一致する頂点が存在する。
    s = 1.0 / math.sqrt(2)  # ≈ 0.7071
    insc_verts = np.array([
        [ s,  s],   # 0: on, param=0.125  (単位円上, 45°)
        [-s,  s],   # 1: on, param=0.375  (単位円上, 135°)
        [-s, -s],   # 2: on, param=0.625  (単位円上, 225°)
        [ s, -s],   # 3: on, param=0.875  (単位円上, 315°)
    ], dtype=float)
    insc_on     = [True, True, True, True]
    insc_params = [0.125, 0.375, 0.625, 0.875]
    inscribed = PolygonData(insc_verts, insc_on, insc_params)

    # --- approximate: 単位円上の n_approx 頂点 (param = k/n_approx) ---
    # n_approx=16 のとき param=0.125 の頂点は (cos(π/4), sin(π/4)) = (s, s)
    # であり inscribed の頂点と数値的に一致する。
    angles       = [2 * math.pi * k / n_approx for k in range(n_approx)]
    approx_verts = np.array([[math.cos(a), math.sin(a)] for a in angles], dtype=float)
    approx_params = [k / n_approx for k in range(n_approx)]
    approximate = PolygonData(approx_verts, [True] * n_approx, approx_params)

    return CurveContainmentPolygons(circumscribed, inscribed, approximate)


# ---------------------------------------------------------------------------
# PolygonData.get_curve_param_index
# ---------------------------------------------------------------------------

class TestGetCurveParamIndex:

    def test_both_endpoints_on_curve(self):
        """両端点が on_curve の場合、端点インデックスをそのまま返す。"""
        poly = _square_polygon()
        assert poly.get_curve_param_index(0) == (0, 1)

    def test_last_edge_on_curve(self):
        """最終辺 (M-1 → 0) が両端 on_curve。"""
        poly = _square_polygon()
        assert poly.get_curve_param_index(3) == (3, 0)

    def test_last_edge_is_wrap(self):
        """最終辺は i_start > i_end となる（折り返し）。"""
        poly = _square_polygon()
        i_start, i_end = poly.get_curve_param_index(3)
        assert i_start > i_end

    def test_start_off_curve_expands_backward(self):
        """始点が off_curve のとき、遡って on_curve を見つける。"""
        # [on, off, on, on]
        verts = np.zeros((4, 2))
        poly = PolygonData(verts, [True, False, True, True], [0.0, 0.0, 0.5, 0.75])
        # edge 1 (off→on): start 側を遡って index 0(on) に到達
        i_start, i_end = poly.get_curve_param_index(1)
        assert i_start == 0
        assert poly.on_curve[i_start]

    def test_end_off_curve_expands_forward(self):
        """終点が off_curve のとき、進んで on_curve を見つける。"""
        # [on, on, off, on]
        verts = np.zeros((4, 2))
        poly = PolygonData(verts, [True, True, False, True], [0.0, 0.25, 0.0, 0.75])
        # edge 1 (on→off): end 側を進んで index 3(on) に到達
        i_start, i_end = poly.get_curve_param_index(1)
        assert i_end == 3
        assert poly.on_curve[i_end]

    def test_both_off_curve_expands_both(self):
        """両端が off_curve のとき、両方向に拡張する。"""
        # [on, off, off, on]
        verts = np.zeros((4, 2))
        poly = PolygonData(verts, [True, False, False, True], [0.0, 0.0, 0.0, 0.75])
        # edge 1 (off→off)
        i_start, i_end = poly.get_curve_param_index(1)
        assert i_start == 0
        assert i_end == 3

    def test_wrap_case_off_curve_start(self):
        """折り返しケース: 始点が off_curve で遡ると末尾 on_curve に到達。"""
        # [on, off, off, off, off, on] (m=6)
        verts = np.zeros((6, 2))
        poly = PolygonData(
            verts,
            [True, False, False, False, False, True],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        )
        # edge 0 (on→off): end 側は index 5(on)
        # edge 0 の start は index 0(on) のまま
        i_start, i_end = poly.get_curve_param_index(0)
        assert i_start == 0
        assert i_end == 5
        # 折り返しではない
        assert not (i_start > i_end)

    def test_wrap_detection_index_comparison(self):
        """i_start > i_end が True のとき折り返しと判定できる。"""
        # [on, off, off, off, off, on] (m=6), edge 4 は折り返し経路
        verts = np.zeros((6, 2))
        poly = PolygonData(
            verts,
            [True, False, False, False, False, True],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        )
        # edge 4 (off→on): start 遡って 0(on), end = 5(on)
        # ここでは折り返し発生しない(0 < 5)
        i_start, i_end = poly.get_curve_param_index(4)
        assert poly.on_curve[i_start]
        assert poly.on_curve[i_end]

    def test_raises_on_negative_edge_index(self):
        poly = _square_polygon()
        with pytest.raises(ValueError):
            poly.get_curve_param_index(-1)

    def test_raises_on_out_of_range_edge_index(self):
        poly = _square_polygon()
        with pytest.raises(ValueError):
            poly.get_curve_param_index(4)

    def test_raises_when_all_off_curve(self):
        verts = np.zeros((3, 2))
        poly = PolygonData(verts, [False, False, False], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            poly.get_curve_param_index(0)


# ---------------------------------------------------------------------------
# _winding_number
# ---------------------------------------------------------------------------

class TestWindingNumber:

    def _ccw_square(self):
        return np.array([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])

    def test_inside_returns_nonzero(self):
        assert _winding_number(np.array([0.5, 0.5]), self._ccw_square()) != 0

    def test_outside_returns_zero(self):
        assert _winding_number(np.array([2.0, 2.0]), self._ccw_square()) == 0

    def test_outside_below_returns_zero(self):
        assert _winding_number(np.array([0.5, -1.0]), self._ccw_square()) == 0

    def test_cw_square_inside_nonzero(self):
        """時計回り多角形でも内部は winding number が非ゼロ。"""
        cw = self._ccw_square()[::-1]
        assert _winding_number(np.array([0.5, 0.5]), cw) != 0

    def test_origin_inside_unit_square(self):
        sq = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        assert _winding_number(np.array([0.0, 0.0]), sq) != 0

    def test_far_outside_unit_square(self):
        sq = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        assert _winding_number(np.array([5.0, 5.0]), sq) == 0


# ---------------------------------------------------------------------------
# _point_segment_dist_sq
# ---------------------------------------------------------------------------

class TestPointSegmentDistSq:

    def test_perpendicular_to_midpoint(self):
        """線分の中点に垂直な点: 距離の二乗が正確に計算される。"""
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 0.0])
        p = np.array([1.0, 3.0])
        assert math.isclose(_point_segment_dist_sq(p, a, b), 9.0)

    def test_closest_to_endpoint_a(self):
        """最近傍が始点 A の場合。"""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = np.array([-2.0, 0.0])
        assert math.isclose(_point_segment_dist_sq(p, a, b), 4.0)

    def test_closest_to_endpoint_b(self):
        """最近傍が終点 B の場合。"""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        p = np.array([3.0, 0.0])
        assert math.isclose(_point_segment_dist_sq(p, a, b), 4.0)

    def test_point_on_segment(self):
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        p = np.array([2.0, 0.0])
        assert math.isclose(_point_segment_dist_sq(p, a, b), 0.0)

    def test_degenerate_segment(self):
        """線分が零ベクトル（a == b）の場合は点 a への距離の二乗を返す。"""
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        p = np.array([4.0, 5.0])
        assert math.isclose(_point_segment_dist_sq(p, a, b), 9.0 + 16.0)


# ---------------------------------------------------------------------------
# _find_nearest_edge
# ---------------------------------------------------------------------------

class TestFindNearestEdge:

    def test_nearest_edge_index(self):
        """点に最も近い辺のインデックスが正しく返る。"""
        poly = _square_polygon()  # 辺: 0→1(右辺), 1→2(上辺), 2→3(左辺), 3→0(下辺)
        # 点 (1.5, 0) は右辺 (index 0) に最も近い
        idx, _ = _find_nearest_edge(np.array([1.5, 0.0]), poly)
        assert idx == 0

    def test_nearest_edge_top(self):
        """点 (0, 1.5) は上辺 (index 1) に最も近い。"""
        poly = _square_polygon()
        idx, _ = _find_nearest_edge(np.array([0.0, 1.5]), poly)
        assert idx == 1

    def test_distance_sq_is_positive(self):
        poly = _square_polygon()
        _, d = _find_nearest_edge(np.array([0.0, 0.0]), poly)
        assert d >= 0.0

    def test_point_on_edge_returns_zero_distance(self):
        poly = _square_polygon()
        # 辺 0 の中点 (1, 0) 上の点
        _, d = _find_nearest_edge(np.array([1.0, 0.0]), poly)
        assert math.isclose(d, 0.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# _extract_vertices_range
# ---------------------------------------------------------------------------

class TestExtractVerticesRange:

    def _poly(self):
        verts = np.array([[float(i), 0.0] for i in range(6)])
        return PolygonData(verts, [True] * 6, [i / 6 for i in range(6)])

    def test_non_wrap_consecutive(self):
        poly = self._poly()
        result = _extract_vertices_range(poly, 1, 3, is_wrap=False)
        expected = np.array([[1., 0.], [2., 0.], [3., 0.]])
        np.testing.assert_array_equal(result, expected)

    def test_non_wrap_single_vertex(self):
        poly = self._poly()
        result = _extract_vertices_range(poly, 2, 2, is_wrap=False)
        np.testing.assert_array_equal(result, np.array([[2., 0.]]))

    def test_non_wrap_full_range(self):
        poly = self._poly()
        result = _extract_vertices_range(poly, 0, 5, is_wrap=False)
        assert len(result) == 6

    def test_wrap_crosses_end(self):
        """折り返し: 末尾から先頭に巻く。"""
        poly = self._poly()
        result = _extract_vertices_range(poly, 4, 1, is_wrap=True)
        expected = np.array([[4., 0.], [5., 0.], [0., 0.], [1., 0.]])
        np.testing.assert_array_equal(result, expected)

    def test_wrap_last_to_first(self):
        poly = self._poly()
        result = _extract_vertices_range(poly, 5, 0, is_wrap=True)
        expected = np.array([[5., 0.], [0., 0.]])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _extract_approx_by_param
# ---------------------------------------------------------------------------

class TestExtractApproxByParam:

    def _approx(self):
        """param = 0.0, 0.1, 0.2, ..., 0.9 の 10 頂点。"""
        params = [i / 10 for i in range(10)]
        verts = np.array([[float(i), 0.0] for i in range(10)])
        return PolygonData(verts, [True] * 10, params)

    def test_non_wrap_subset(self):
        approx = self._approx()
        result = _extract_approx_by_param(approx, 0.2, 0.4, is_wrap=False)
        # param 0.2, 0.3, 0.4 → indices 2, 3, 4
        assert len(result) == 3
        np.testing.assert_array_equal(result[:, 0], [2., 3., 4.])

    def test_non_wrap_exact_boundaries(self):
        approx = self._approx()
        result = _extract_approx_by_param(approx, 0.0, 0.9, is_wrap=False)
        assert len(result) == 10

    def test_non_wrap_single_vertex(self):
        approx = self._approx()
        result = _extract_approx_by_param(approx, 0.5, 0.5, is_wrap=False)
        assert len(result) == 1
        assert result[0, 0] == 5.0

    def test_wrap_crosses_boundary(self):
        """折り返し: param_start=0.7, param_end=0.2 → 0.7,0.8,0.9 + 0.0,0.1,0.2"""
        approx = self._approx()
        result = _extract_approx_by_param(approx, 0.7, 0.2, is_wrap=True)
        assert len(result) == 6
        np.testing.assert_array_equal(result[:, 0], [7., 8., 9., 0., 1., 2.])

    def test_wrap_preserves_order(self):
        """折り返し結果は param_start 側が先頭になる。"""
        approx = self._approx()
        result = _extract_approx_by_param(approx, 0.8, 0.1, is_wrap=True)
        # 0.8, 0.9, 0.0, 0.1
        np.testing.assert_array_equal(result[:, 0], [8., 9., 0., 1.])


# ---------------------------------------------------------------------------
# is_point_in_polygon
# ---------------------------------------------------------------------------
#
# フィクスチャ: 単位円を近似した CurveContainmentPolygons (_make_circle_polygons)
#   circumscribed: 外接正方形（辺 x=±1, y=±1）
#     接点 (on_curve=True): (1,0),(0,1),(-1,0),(0,-1) → param 0, 0.25, 0.5, 0.75
#     角頂点 (on_curve=False): (1,1),(-1,1),(-1,-1),(1,-1)
#   inscribed: 内接正方形（45° 回転, 辺 x=±s, y=±s, s=1/√2≈0.707）
#     全頂点 on_curve=True、単位円上: (s,s),(-s,s),(-s,-s),(s,-s)
#     → param 0.125, 0.375, 0.625, 0.875
#   approximate: 単位円上の 16 角形
#     param = k/16 (k=0..15)
#
# アルゴリズムの分岐:
#   Step 1: 外包多角形（x=±1, y=±1 の正方形）の外部 → False
#   Step 2: 内包多角形（|x|≤s AND |y|≤s の正方形）の内部 → True
#   Step 3: 灰色地帯（内包/外包の間）→ 再構成多角形で判定
#     3a. 内包多角形の辺が最近傍 → 反転なし
#     3b. 外包多角形の辺が最近傍 → 反転あり（is_inscribed=False → return not inside）
#
# 折り返しの発生条件:
#   内包辺 3 (s,-s)→(s,s): param 0.875→0.125 → is_wrap=True
#   外包辺 7 (off)→(1,0): get_curve_param_index → (6,0), is_wrap=True

class TestIsPointInPolygon:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.polygons = _make_circle_polygons(n_approx=16)
        self.s = 1.0 / math.sqrt(2)  # ≈ 0.7071 (内接正方形の辺座標)

    # -----------------------------------------------------------------------
    # Step 1: 外包多角形の外部 → False
    # -----------------------------------------------------------------------

    def test_step1_far_outside_x_axis(self):
        """x 軸方向に明らかに遠い点。"""
        assert is_point_in_polygon(np.array([3.0, 0.0]), self.polygons) is False

    def test_step1_far_outside_y_axis(self):
        """y 軸方向に明らかに遠い点。"""
        assert is_point_in_polygon(np.array([0.0, 2.0]), self.polygons) is False

    def test_step1_outside_diagonal(self):
        """対角線方向に遠い点（外包多角形の角の外側）。"""
        assert is_point_in_polygon(np.array([1.5, 1.5]), self.polygons) is False

    def test_step1_just_outside_circumscribed_edge(self):
        """外包多角形の辺を僅かに超えた点 (x=1.01)。"""
        assert is_point_in_polygon(np.array([1.01, 0.0]), self.polygons) is False

    def test_step1_just_outside_on_y(self):
        """外包多角形の上辺を僅かに超えた点 (y=1.01)。"""
        assert is_point_in_polygon(np.array([0.0, 1.01]), self.polygons) is False

    # -----------------------------------------------------------------------
    # Step 2: 内包多角形の内部 → True
    # -----------------------------------------------------------------------

    def test_step2_origin(self):
        """原点は内包多角形の内部。"""
        assert is_point_in_polygon(np.array([0.0, 0.0]), self.polygons) is True

    def test_step2_interior_point(self):
        """内包多角形の深い内部。"""
        assert is_point_in_polygon(np.array([0.3, 0.3]), self.polygons) is True

    def test_step2_just_inside_inscribed_edge(self):
        """内包多角形の右辺 x=s の直内 (x=0.70 < s≈0.707)。"""
        assert is_point_in_polygon(np.array([0.70, 0.0]), self.polygons) is True

    def test_step2_negative_quadrant(self):
        """第三象限の内包多角形内部。"""
        assert is_point_in_polygon(np.array([-0.3, -0.3]), self.polygons) is True

    # -----------------------------------------------------------------------
    # Step 3a: 内包多角形の辺が最近傍 / 折り返しなし
    #   内包辺 0: (s,s)→(-s,s)  param 0.125→0.375  is_wrap=False
    #   内包辺 2: (-s,-s)→(s,-s) param 0.625→0.875  is_wrap=False
    # -----------------------------------------------------------------------

    def test_step3a_nowrap_above_inscribed_top_inside(self):
        """内包多角形の上辺付近、単位円内 (r=0.75 < 1)。
        最近傍: 内包辺 0 (y=s≈0.707)、is_wrap=False。"""
        # y=0.75 > s=0.707 → 内包多角形外。dist to insc_edge0 ≈ 0.043
        # dist to circ_edge2 (y=1) ≈ 0.25 → 内包が最近傍
        assert is_point_in_polygon(np.array([0.0, 0.75]), self.polygons) is True

    def test_step3a_nowrap_above_inscribed_top_outside(self):
        """内包多角形の上辺付近、単位円外 (r≈1.016)。
        最近傍: 内包辺 0 (y=s)、is_wrap=False。期待値: False。
        点 (0.6, 0.82): r=√(0.36+0.6724)≈1.016>1、y=0.82>s≈0.707（内包外）
        dist_insc0=0.82-0.707=0.113 < dist_circ=0.18 → 内包最近傍。"""
        assert is_point_in_polygon(np.array([0.6, 0.82]), self.polygons) is False

    def test_step3a_nowrap_below_inscribed_bottom_inside(self):
        """内包多角形の下辺付近、単位円内 (r=0.75 < 1)。
        最近傍: 内包辺 2 (y=-s)、is_wrap=False。"""
        assert is_point_in_polygon(np.array([0.0, -0.75]), self.polygons) is True

    def test_step3a_nowrap_below_inscribed_bottom_outside(self):
        """内包多角形の下辺付近、単位円外 (r≈1.016)。
        最近傍: 内包辺 2 (y=-s)、is_wrap=False。期待値: False。
        点 (-0.6, -0.82): 上辺外側ケースの対称点。"""
        assert is_point_in_polygon(np.array([-0.6, -0.82]), self.polygons) is False

    # -----------------------------------------------------------------------
    # Step 3a: 内包多角形の辺が最近傍 / 折り返しあり
    #   内包辺 3: (s,-s)→(s,s)  param 0.875→0.125  is_wrap=True
    #   get_curve_param_index(3) → (3, 0), 3 > 0
    # -----------------------------------------------------------------------

    def test_step3a_wrap_right_of_inscribed_inside(self):
        """内包多角形の右辺付近 (x=0.75)、単位円内。
        最近傍: 内包辺 3 (x=s)、is_wrap=True。期待値: True。"""
        # dist to insc_edge3 (x=s): 0.75-0.707≈0.043
        # dist to circ_edge0 (x=1): 1-0.75=0.25 → 内包が最近傍
        assert is_point_in_polygon(np.array([0.75, 0.0]), self.polygons) is True

    def test_step3a_wrap_right_of_inscribed_outside(self):
        """内包多角形の右辺付近、単位円外 (r>1 かつ内包辺が最近傍)。
        最近傍: 内包辺 3、is_wrap=True。期待値: False。"""
        # (0.75, -0.8): r²=0.5625+0.64=1.2025>1
        # dist to insc_edge3 (x=s, y in [-s,s]): closest (s,-s), dist=√((0.75-s)²+(0.8-s)²)
        assert is_point_in_polygon(np.array([0.75, -0.8]), self.polygons) is False

    def test_step3a_wrap_right_of_inscribed_inside_lower(self):
        """内包多角形の右辺付近（下方）、単位円内。
        最近傍: 内包辺 3、is_wrap=True。"""
        # (0.85, -0.3): r²=0.7225+0.09=0.8125<1
        assert is_point_in_polygon(np.array([0.85, -0.3]), self.polygons) is True

    # -----------------------------------------------------------------------
    # Step 3b: 外包多角形の辺が最近傍 / 折り返しなし
    #   外包辺 0: (1,0)→(1,1) off  → get_curve_param_index(0) → (0,2), is_wrap=False
    #   外包辺 2: (0,1)→(-1,1) off → get_curve_param_index(2) → (2,4), is_wrap=False
    # -----------------------------------------------------------------------

    def test_step3b_nowrap_right_circ_inside(self):
        """外包多角形の右辺付近 (x=0.9)、単位円内。
        最近傍: 外包辺 0 (x=1)、is_wrap=False。期待値: True（反転ロジック）。"""
        # dist to circ_edge0 (x=1): 0.1  /  dist to insc_edge3 (x=s): 0.9-0.707≈0.193
        assert is_point_in_polygon(np.array([0.9, 0.0]), self.polygons) is True

    def test_step3b_nowrap_corner_area_outside(self):
        """外包多角形の角付近 (0.95, 0.95)、単位円外 (r≈1.34)。
        最近傍: 外包辺 0 か辺 1（どちらも x=1 または y=1）、is_wrap=False。期待値: False。"""
        assert is_point_in_polygon(np.array([0.95, 0.95]), self.polygons) is False

    def test_step3b_nowrap_upper_right_inside(self):
        """外包多角形の上辺付近 (0.0, 0.9)、単位円内。
        最近傍: 外包辺 2 (y=1)、is_wrap=False。期待値: True。"""
        assert is_point_in_polygon(np.array([0.0, 0.9]), self.polygons) is True

    def test_step3b_nowrap_upper_right_outside(self):
        """外包多角形の上辺付近、単位円外かつ外包内 (0.5, 0.9)、r≈1.03。
        最近傍: 外包辺 2 (y=1)。期待値: False。"""
        # r = sqrt(0.25+0.81) = sqrt(1.06) ≈ 1.03 > 1
        assert is_point_in_polygon(np.array([0.5, 0.9]), self.polygons) is False

    # -----------------------------------------------------------------------
    # Step 3b: 外包多角形の辺が最近傍 / 折り返しあり
    #   外包辺 7: (1,-1)[off]→(1,0)[on]  → get_curve_param_index(7) → (6,0), is_wrap=True
    #   外包辺 6: (0,-1)[on]→(1,-1)[off] → get_curve_param_index(6) → (6,0), is_wrap=True
    # -----------------------------------------------------------------------

    def test_step3b_wrap_lower_right_circ_inside(self):
        """外包多角形の右辺下半分付近、単位円内。
        最近傍: 外包辺 7 (x=1, y<0)、is_wrap=True。期待値: True（反転ロジック）。"""
        # (0.88, -0.4): r²=0.7744+0.16=0.9344<1
        # dist to circ_edge7 (x=1, y in [-1,0]): 1-0.88=0.12
        # dist to insc_edge3 (x=s, y in [-s,s]): 0.88-0.707≈0.173 → 外包が最近傍
        assert is_point_in_polygon(np.array([0.88, -0.4]), self.polygons) is True

    def test_step3b_wrap_lower_right_circ_outside(self):
        """外包多角形の右辺下半分付近、単位円外。
        最近傍: 外包辺 7、is_wrap=True。期待値: False（反転ロジック）。"""
        # (0.98, -0.5): r²=0.9604+0.25=1.2104>1
        assert is_point_in_polygon(np.array([0.98, -0.5]), self.polygons) is False

    # -----------------------------------------------------------------------
    # 数値安定性: 曲線境界付近
    # -----------------------------------------------------------------------

    def test_numerical_barely_inside_x_axis(self):
        """単位円の直内 (r=0.999)。"""
        assert is_point_in_polygon(np.array([0.999, 0.0]), self.polygons) is True

    def test_numerical_barely_inside_diagonal(self):
        """45° 方向の直内 (r=0.999)。"""
        r = 0.999
        assert is_point_in_polygon(
            np.array([r / math.sqrt(2), r / math.sqrt(2)]), self.polygons) is True

    def test_numerical_barely_outside_diagonal(self):
        """45° 方向の直外 (r=1.001/√2≈0.708)、外包多角形内。"""
        # r=1.001: r/√2≈0.7078、x<1 かつ y<1 なので外包内
        r = 1.001
        assert is_point_in_polygon(
            np.array([r / math.sqrt(2), r / math.sqrt(2)]), self.polygons) is False

    # -----------------------------------------------------------------------
    # 各象限の対称性確認
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("x, y", [
        ( 0.9,  0.0),  # 第1象限（+x 軸）
        ( 0.0,  0.9),  # 第1象限（+y 軸）
        (-0.9,  0.0),  # 第2/3象限（-x 軸）
        ( 0.0, -0.9),  # 第3/4象限（-y 軸）
    ])
    def test_gray_zone_inside_all_quadrants(self, x, y):
        """灰色地帯の単位円内点が全方向で True になる。"""
        assert is_point_in_polygon(np.array([x, y]), self.polygons) is True

    @pytest.mark.parametrize("x, y", [
        ( 0.95,  0.95),  # 第1象限の角
        (-0.95,  0.95),  # 第2象限の角
        (-0.95, -0.95),  # 第3象限の角
        ( 0.95, -0.95),  # 第4象限の角
    ])
    def test_corner_area_outside_all_quadrants(self, x, y):
        """外包多角形の各角付近（単位円外）が全象限で False になる。"""
        assert is_point_in_polygon(np.array([x, y]), self.polygons) is False

    # -----------------------------------------------------------------------
    # 境界点のロバスト性: クラッシュしないこと
    # -----------------------------------------------------------------------

    def test_robustness_on_approximate_polygon_vertex(self):
        """近似多角形の頂点上（曲線境界）でクラッシュしない。"""
        p = np.array([1.0, 0.0])  # approx param=0.0 の頂点
        result = is_point_in_polygon(p, self.polygons)
        assert isinstance(result, bool)

    def test_robustness_on_inscribed_vertex(self):
        """内包多角形の頂点上でクラッシュしない。"""
        p = np.array([self.s, self.s])  # inscribed vertex[0]
        result = is_point_in_polygon(p, self.polygons)
        assert isinstance(result, bool)

    def test_robustness_on_circumscribed_on_curve_vertex(self):
        """外包多角形の on_curve 頂点上でクラッシュしない。"""
        p = np.array([1.0, 0.0])  # circ vertex[0] = approx vertex at param=0
        result = is_point_in_polygon(p, self.polygons)
        assert isinstance(result, bool)
