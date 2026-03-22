"""algorithms.py / _alg_polygonal_approximation.py の結合テスト"""
import pytest
import numpy as np

from py_igesio.entities.curves import EllipseCurve, PolylineCurve2D
from py_igesio.entities.curves._alg_extremal_polygon import _compute_extremal_polygon
from py_igesio.entities.curves._alg_polygonal_approximation import compute_approximate_polygon
from py_igesio.entities.curves.algorithms import compute_containment_polygons
from py_igesio.numerics.polygons import CurveContainmentPolygons


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
# compute_approximate_polygon
# -----------------------------------------------------------------------

class TestComputeApproximatePolygon:
    def test_all_on_curve(self, ellipse):
        """近似多角形の全頂点が on_curve=True"""
        inscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        circumscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        result = compute_approximate_polygon(ellipse, inscribed, circumscribed, eps=0.01)
        assert all(result.on_curve)

    def test_fixed_points_included(self, ellipse):
        """内包・外包多角形の on_curve=True パラメータがすべて近似多角形に含まれる"""
        inscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        circumscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        result = compute_approximate_polygon(ellipse, inscribed, circumscribed, eps=0.01)

        t_min, t_max = ellipse.get_range()
        fixed = set()
        for pd in (inscribed, circumscribed):
            for u, on in zip(pd.curve_params, pd.on_curve):
                if on and not np.isclose(u, t_max):
                    fixed.add(u)
        fixed.add(t_min)

        for u in fixed:
            assert u in result.curve_params

    def test_coordinate_consistency(self, ellipse):
        """全頂点で curve.evaluate(curve_params[i]) と座標が一致"""
        inscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        circumscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        result = compute_approximate_polygon(ellipse, inscribed, circumscribed, eps=0.01)
        for v, u in zip(result.vertices, result.curve_params):
            np.testing.assert_allclose(v, ellipse.evaluate(u), atol=1e-12)

    def test_smaller_eps_yields_more_points(self, ellipse):
        """eps を小さくすると点数が増加（または同数）"""
        inscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=False)
        circumscribed = _compute_extremal_polygon(ellipse, n_vert=4, circumscribed=True)
        r_coarse = compute_approximate_polygon(ellipse, inscribed, circumscribed, eps=0.1)
        r_fine = compute_approximate_polygon(ellipse, inscribed, circumscribed, eps=0.001)
        assert r_fine.count() >= r_coarse.count()


# -----------------------------------------------------------------------
# compute_containment_polygons
# -----------------------------------------------------------------------

class TestComputeContainmentPolygons:
    def test_returns_curve_containment_polygons(self, ellipse):
        result = compute_containment_polygons(ellipse, n_vert=4)
        assert isinstance(result, CurveContainmentPolygons)

    def test_all_polygons_nonempty(self, ellipse):
        result = compute_containment_polygons(ellipse, n_vert=4)
        assert result.circumscribed.count() > 0
        assert result.inscribed.count() > 0
        assert result.approximate.count() > 0

    def test_approximate_all_on_curve(self, ellipse):
        result = compute_containment_polygons(ellipse, n_vert=4)
        assert all(result.approximate.on_curve)

    def test_ellipse_geometric_consistency(self, ellipse):
        """楕円のパイプライン全体にわたる幾何的整合性を検証する

        - 内包多角形の on_curve 頂点は楕円上にある
        - 外包多角形の off-curve 頂点（接線交点）は楕円の外側にある
        - 近似多角形の全頂点座標が curve.evaluate と一致する
        """
        result = compute_containment_polygons(ellipse, n_vert=4)
        a, b = 2.0, 1.0

        for v, on in zip(result.inscribed.vertices, result.inscribed.on_curve):
            if on:
                assert pytest.approx((v[0] / a) ** 2 + (v[1] / b) ** 2, abs=1e-6) == 1.0

        for v, on in zip(result.circumscribed.vertices, result.circumscribed.on_curve):
            if not on:
                assert (v[0] / a) ** 2 + (v[1] / b) ** 2 >= 1.0 - 1e-9

        for v, u in zip(result.approximate.vertices, result.approximate.curve_params):
            np.testing.assert_allclose(v, ellipse.evaluate(u), atol=1e-12)

    def test_approximate_contains_fixed_points(self, ellipse):
        """内包・外包多角形の on_curve=True パラメータが近似多角形に含まれる"""
        result = compute_containment_polygons(ellipse, n_vert=4)
        t_min, t_max = ellipse.get_range()

        fixed = set()
        for pd in (result.inscribed, result.circumscribed):
            for u, on in zip(pd.curve_params, pd.on_curve):
                if on and not np.isclose(u, t_max):
                    fixed.add(u)
        fixed.add(t_min)

        for u in fixed:
            assert u in result.approximate.curve_params

    def test_polyline_corners_in_polygons(self, unit_square):
        """外包・内包多角形の curve_params に角点 0,1,2,3 が含まれる"""
        result = compute_containment_polygons(unit_square, n_vert=4)
        for corner in [0.0, 1.0, 2.0, 3.0]:
            assert corner in result.circumscribed.curve_params
            assert corner in result.inscribed.curve_params
