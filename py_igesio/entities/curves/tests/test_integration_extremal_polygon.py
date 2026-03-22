"""_alg_extremal_polygon.py の結合テスト"""
import math

import pytest
import numpy as np

from py_igesio.entities.curves import EllipseCurve, PolylineCurve2D
from py_igesio.entities.curves._alg_extremal_polygon import (
    _sample_points,
    _build_polygon_vertices,
    _compute_extremal_polygon,
    _in_linear,
)


# -----------------------------------------------------------------------
# フィクスチャ
# -----------------------------------------------------------------------

@pytest.fixture
def ellipse():
    """a=2, b=1 の楕円曲線（CCW・曲率極値あり）"""
    return EllipseCurve(a=2.0, b=1.0)


@pytest.fixture
def unit_square():
    """CCW の単位正方形ポリライン（角点・直線部あり）"""
    return PolylineCurve2D([(0, 0), (1, 0), (1, 1), (0, 1)])


# -----------------------------------------------------------------------
# _sample_points
# -----------------------------------------------------------------------

class TestSamplePoints:
    def test_ts_pts_same_length(self, ellipse):
        ts, pts = _sample_points(ellipse, 8)
        assert len(ts) == len(pts)

    def test_ts_is_sorted(self, ellipse):
        ts, _ = _sample_points(ellipse, 8)
        assert ts == sorted(ts)

    def test_pts_match_curve_evaluate(self, ellipse):
        ts, pts = _sample_points(ellipse, 8)
        for t, pt in zip(ts, pts):
            np.testing.assert_allclose(pt, ellipse.evaluate(t), atol=1e-12)

    def test_curvature_extrema_included(self, ellipse):
        """EllipseCurve(a=2, b=1): 曲率極値 t≈0, π/2, π, 3π/2 が ts に含まれる"""
        ts, _ = _sample_points(ellipse, 4)
        period = 2 * math.pi
        for expected in [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            assert any(
                min(abs(t - expected), period - abs(t - expected)) < 1e-2
                for t in ts
            )

    def test_corners_included(self, unit_square):
        """PolylineCurve2D の角点パラメータがすべて ts に含まれる"""
        ts, _ = _sample_points(unit_square, 4)
        for corner in [0.0, 1.0, 2.0, 3.0]:
            assert corner in ts

    def test_no_interior_linear_points(self, unit_square):
        """直線部内部にはサンプル点が落ちない"""
        ts, _ = _sample_points(unit_square, 8)
        linear_segments = unit_square.get_linear_segments()
        for t in ts:
            assert not _in_linear(linear_segments, t)


# -----------------------------------------------------------------------
# _build_polygon_vertices
# -----------------------------------------------------------------------

class TestBuildPolygonVertices:
    # 楕円の曲率極値に対応する固定 ts を使用
    _TS = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]

    def _ts_pts(self, ellipse):
        ts = list(self._TS)
        pts = [ellipse.evaluate(t) for t in ts]
        return ts, pts

    def test_all_vertex_no_extra_points(self, ellipse):
        """is_contact がすべて False のとき交点は挿入されず len(ts) 個の頂点のみ"""
        ts, pts = self._ts_pts(ellipse)
        is_contact = [False] * len(ts)
        result = _build_polygon_vertices(ellipse, ts, pts, is_contact)
        assert result.count() == len(ts)
        assert all(result.on_curve)

    def test_all_contact_doubles_count(self, ellipse):
        """is_contact がすべて True のとき各区間に接線交点が挿入されて頂点数が 2 倍"""
        ts, pts = self._ts_pts(ellipse)
        is_contact = [True] * len(ts)
        result = _build_polygon_vertices(ellipse, ts, pts, is_contact)
        assert result.count() == 2 * len(ts)

    def test_all_contact_on_curve_alternates(self, ellipse):
        """接点(True) → 交点(False) → 接点(True) → … と交互になる"""
        ts, pts = self._ts_pts(ellipse)
        is_contact = [True] * len(ts)
        result = _build_polygon_vertices(ellipse, ts, pts, is_contact)
        for i, on in enumerate(result.on_curve):
            assert on == (i % 2 == 0)

    def test_on_curve_vertices_coordinate_consistency(self, ellipse):
        """on_curve=True の全頂点で curve.evaluate(curve_params[i]) と座標が一致"""
        ts, pts = self._ts_pts(ellipse)
        is_contact = [True] * len(ts)
        result = _build_polygon_vertices(ellipse, ts, pts, is_contact)
        for v, on, param in zip(result.vertices, result.on_curve, result.curve_params):
            if on:
                np.testing.assert_allclose(v, ellipse.evaluate(param), atol=1e-12)

    def test_linear_pair_suppresses_intersection(self, unit_square):
        """直線区間ペアには交点が追加されず、非直線ペア分の交点のみ生成される

        ts=[0,1,2,3]: ペア (0,1),(1,2),(2,3) は直線ペア → Pa のみ (計3頂点)
                      ペア (3,0) は非直線ペア、両接点 → Pa + 交点 (計2頂点)
        合計: 5頂点、on_curve=False は 1 個
        """
        ts = [0.0, 1.0, 2.0, 3.0]
        pts = [unit_square.evaluate(t) for t in ts]
        is_contact = [True] * len(ts)
        result = _build_polygon_vertices(unit_square, ts, pts, is_contact)
        assert result.count() == 5
        assert result.on_curve.count(False) == 1


# -----------------------------------------------------------------------
# _compute_extremal_polygon
# -----------------------------------------------------------------------

class TestComputeExtremalPolygon:
    def test_returns_nonempty_polygon(self, ellipse):
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        assert result.count() > 0

    def test_data_arrays_consistent_length(self, ellipse):
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        assert len(result.on_curve) == result.count()
        assert len(result.curve_params) == result.count()

    def test_on_curve_vertices_coordinate_consistency(self, ellipse):
        """on_curve=True の頂点は curve.evaluate(curve_params[i]) と一致"""
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        for v, on, param in zip(result.vertices, result.on_curve, result.curve_params):
            if on:
                np.testing.assert_allclose(v, ellipse.evaluate(param), atol=1e-12)

    def test_circumscribed_off_curve_outside_ellipse(self, ellipse):
        """外包多角形の off-curve 頂点（接線交点）は楕円の外側にある"""
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        a, b = 2.0, 1.0
        for v, on in zip(result.vertices, result.on_curve):
            if not on:
                assert (v[0] / a) ** 2 + (v[1] / b) ** 2 >= 1.0 - 1e-9

    def test_inscribed_all_on_curve(self, ellipse):
        """楕円は全域で外側凸なので内包多角形の全頂点が on_curve=True"""
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        assert all(result.on_curve)

    def test_inscribed_vertices_on_ellipse(self, ellipse):
        """内包多角形の全頂点が楕円上にある"""
        result = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        a, b = 2.0, 1.0
        for v in result.vertices:
            assert pytest.approx((v[0] / a) ** 2 + (v[1] / b) ** 2, abs=1e-6) == 1.0

    def test_polyline_corners_in_curve_params(self, unit_square):
        """外包多角形の curve_params に角点パラメータ 0,1,2,3 が含まれる"""
        result = _compute_extremal_polygon(unit_square, n_vert=4, circumscribed=True)
        for corner in [0.0, 1.0, 2.0, 3.0]:
            assert corner in result.curve_params
