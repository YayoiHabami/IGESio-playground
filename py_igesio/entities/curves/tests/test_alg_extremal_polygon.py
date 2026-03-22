"""_alg_extremal_polygon.py の単体テスト"""
import math

import pytest
import numpy as np

from py_igesio.entities.interfaces import CurveBase2D
from py_igesio.entities.curves import EllipseCurve, StarCurve, PolylineCurve2D
from py_igesio.entities.curves._alg_extremal_polygon import (
    _line_intersect,
    _find_zero_curvature,
    _is_convex_outward,
    _is_convex_inward,
    _validate_curve,
    _dedup,
    _find_curvature_extrema,
    _interpolate_closed_points,
    _insert_corners,
    _insert_linear_endpoints,
    _in_linear,
    _is_linear_interval,
    _classify_points,
)


class _MockOpenCurve(CurveBase2D):
    """テスト用: is_closed()=False の最小実装"""
    def get_name(self): return "OpenMock"
    def is_closed(self): return False
    def has_self_intersection(self): return False
    def is_clockwise(self): return False
    def get_range(self): return (0.0, 1.0)
    def evaluate(self, t): return np.array([t, 0.0])
    def derivative(self, t, order): return np.array([1.0, 0.0])


# -----------------------------------------------------------------------
# フィクスチャ
# -----------------------------------------------------------------------

@pytest.fixture
def ellipse():
    """a=2, b=1 の楕円曲線（角点なし・CCW）"""
    return EllipseCurve(a=2.0, b=1.0)


@pytest.fixture
def star():
    """r=5, a=0.8, n=5 の星形曲線（CCW）"""
    return StarCurve(r=5.0, a=0.8, n=5)


@pytest.fixture
def unit_square():
    """CCW の単位正方形ポリライン"""
    return PolylineCurve2D([(0, 0), (1, 0), (1, 1), (0, 1)])


# -----------------------------------------------------------------------
# _line_intersect
# -----------------------------------------------------------------------

class TestLineIntersect:
    def test_axis_intersection_at_origin(self):
        """x軸と y軸の交点は原点"""
        pt = _line_intersect(
            np.array([1.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([0.0, 1.0]),
        )
        np.testing.assert_allclose(pt, [0.0, 0.0], atol=1e-10)

    def test_diagonal_lines_intersection(self):
        """y=x と y=1-x の交点は (0.5, 0.5)"""
        pt = _line_intersect(
            np.array([0.0, 0.0]), np.array([1.0, 1.0]),
            np.array([1.0, 0.0]), np.array([1.0, -1.0]),
        )
        np.testing.assert_allclose(pt, [0.5, 0.5], atol=1e-10)

    def test_parallel_lines_return_midpoint(self):
        """平行な直線は中点を返す"""
        pt = _line_intersect(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
        )
        np.testing.assert_allclose(pt, [0.0, 0.5], atol=1e-10)


# -----------------------------------------------------------------------
# _is_convex_outward / _is_convex_inward
# -----------------------------------------------------------------------

class TestIsConvex:
    def test_ellipse_always_outward(self, ellipse):
        """楕円（CCW・κ>0）は常に外側に凸"""
        for t in [0.1, 0.5, 1.0, 2.0, 3.0]:
            assert _is_convex_outward(ellipse, t) is True

    def test_ellipse_never_inward(self, ellipse):
        for t in [0.1, 0.5, 1.0, 2.0, 3.0]:
            assert _is_convex_inward(ellipse, t) is False

    def test_star_tip_is_outward(self, star):
        """星形の先端 (t=0) は外側に凸（κ>0・CCW）"""
        assert _is_convex_outward(star, 0.0) is True
        assert _is_convex_inward(star, 0.0) is False

    def test_star_valley_is_inward(self, star):
        """星形の谷 (t=π/5) は内側に凸（κ<0・CCW）"""
        assert _is_convex_inward(star, math.pi / 5) is True
        assert _is_convex_outward(star, math.pi / 5) is False


# -----------------------------------------------------------------------
# _validate_curve
# -----------------------------------------------------------------------

class TestValidateCurve:
    def test_valid_curve_no_exception(self, ellipse):
        _validate_curve(ellipse)

    def test_open_curve_raises(self):
        with pytest.raises(ValueError):
            _validate_curve(_MockOpenCurve())

    def test_self_intersecting_curve_raises(self):
        with pytest.raises(ValueError):
            _validate_curve(StarCurve(r=0.5, a=1.0))


# -----------------------------------------------------------------------
# _dedup
# -----------------------------------------------------------------------

class TestDedup:
    def test_empty_list(self, ellipse):
        assert _dedup(ellipse, [], 2 * math.pi, 1e-4) == []

    def test_no_duplicates_preserved(self, ellipse):
        pts = [0.0, 1.0, 2.0, 3.0]
        assert _dedup(ellipse, pts, 2 * math.pi, 1e-4) == pts

    def test_close_points_merged(self, ellipse):
        """周期の 1e-5 倍以下の間隔は merge_tol=1e-4 で除去"""
        period = 2 * math.pi
        pts = [1.0, 1.0 + period * 1e-5]
        assert len(_dedup(ellipse, pts, period, 1e-4)) == 1

    def test_far_enough_points_kept(self, ellipse):
        """周期の 1e-3 倍の間隔は merge_tol=1e-4 より大きいので保持"""
        period = 2 * math.pi
        pts = [1.0, 1.0 + period * 1e-3]
        assert len(_dedup(ellipse, pts, period, 1e-4)) == 2

    def test_circular_wrap_dedup(self, ellipse):
        """先頭と末尾が周期的に近い場合、末尾を除去"""
        period = 2 * math.pi
        pts = [period * 1e-5, 1.0, 2.0, period - period * 1e-5]
        assert len(_dedup(ellipse, pts, period, 1e-4)) == 3


# -----------------------------------------------------------------------
# _in_linear
# -----------------------------------------------------------------------

class TestInLinear:
    def test_inside_segment_is_true(self):
        assert _in_linear([(0.0, 2.0)], 1.0) is True

    def test_at_start_endpoint_is_false(self):
        assert _in_linear([(0.0, 2.0)], 0.0) is False

    def test_at_end_endpoint_is_false(self):
        assert _in_linear([(0.0, 2.0)], 2.0) is False

    def test_outside_segment_is_false(self):
        assert _in_linear([(0.0, 2.0)], 3.0) is False

    def test_empty_segments_always_false(self):
        assert _in_linear([], 1.0) is False


# -----------------------------------------------------------------------
# _is_linear_interval
# -----------------------------------------------------------------------

class TestIsLinearInterval:
    def test_exact_match_is_true(self):
        segs = [(0.0, 1.0), (2.0, 3.0)]
        assert _is_linear_interval(segs, 0.0, 1.0) is True
        assert _is_linear_interval(segs, 2.0, 3.0) is True

    def test_within_eps_is_true(self):
        segs = [(0.0, 1.0)]
        assert _is_linear_interval(segs, 1e-10, 1.0 - 1e-10) is True

    def test_wrong_interval_is_false(self):
        segs = [(0.0, 1.0)]
        assert _is_linear_interval(segs, 0.0, 2.0) is False

    def test_empty_segments_is_false(self):
        assert _is_linear_interval([], 0.0, 1.0) is False


# -----------------------------------------------------------------------
# _insert_corners
# -----------------------------------------------------------------------

class TestInsertCorners:
    def test_smooth_curve_unchanged(self, ellipse):
        ts = [0.1, 0.5, 1.0]
        assert _insert_corners(ellipse, ts.copy()) == ts

    def test_corners_inserted_and_sorted(self, unit_square):
        ts = [0.5, 1.5, 2.5, 3.5]
        result = _insert_corners(unit_square, ts.copy())
        for i in range(4):
            assert float(i) in result
        assert result == sorted(result)

    def test_existing_corners_not_duplicated(self, unit_square):
        ts = [0.0, 0.5, 1.0]
        result = _insert_corners(unit_square, ts.copy())
        assert result.count(0.0) == 1
        assert result.count(1.0) == 1


# -----------------------------------------------------------------------
# _insert_linear_endpoints
# -----------------------------------------------------------------------

class TestInsertLinearEndpoints:
    def test_empty_segments_unchanged(self):
        ts = [0.1, 0.5, 1.0]
        assert _insert_linear_endpoints([], ts.copy()) == ts

    def test_endpoints_inserted_and_sorted(self):
        ts = [0.5]
        result = _insert_linear_endpoints([(0.0, 1.0)], ts.copy())
        assert result == [0.0, 0.5, 1.0]

    def test_existing_endpoint_not_duplicated(self):
        ts = [0.0, 0.5]
        result = _insert_linear_endpoints([(0.0, 1.0)], ts.copy())
        assert result.count(0.0) == 1
        assert 1.0 in result


# -----------------------------------------------------------------------
# _interpolate_closed_points
# -----------------------------------------------------------------------

class TestInterpolateClosedPoints:
    def test_empty_input_returns_empty(self):
        assert _interpolate_closed_points([], 0.0, 2 * math.pi, 10) == []

    def test_sufficient_points_unchanged(self):
        """n_vert <= num_pts の場合は変更なし"""
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert _interpolate_closed_points(ts, 0.0, 5.0, 3) == ts

    def test_interpolation_increases_point_count(self):
        """2点を n_vert=8 で補間すると 8点になる"""
        ts = [0.0, math.pi]
        result = _interpolate_closed_points(ts, 0.0, 2 * math.pi, 8)
        assert len(result) == 8

    def test_skip_interval_fn_reduces_points(self):
        """skip_interval_fn=True の区間は始端のみ追加され点数が減る"""
        ts = [0.0, 1.0, 2.0]
        t0, t1, n_vert = 0.0, 3.0, 9
        result_full = _interpolate_closed_points(ts.copy(), t0, t1, n_vert)
        result_skip = _interpolate_closed_points(
            ts.copy(), t0, t1, n_vert,
            skip_interval_fn=lambda ta, tb: (ta == 0.0 and tb == 1.0),
        )
        assert len(result_skip) < len(result_full)
        assert 0.0 in result_skip


# -----------------------------------------------------------------------
# _classify_points
# -----------------------------------------------------------------------

class TestClassifyPoints:
    def test_ellipse_circumscribed_all_contact(self, ellipse):
        """楕円（外側凸のみ）は外包構築で全点が接点"""
        ts = np.linspace(0.1, 2 * math.pi - 0.1, 8).tolist()
        assert all(_classify_points(ellipse, ts, circumscribed=True))

    def test_ellipse_inscribed_all_vertex(self, ellipse):
        """楕円は内包構築で全点が頂点"""
        ts = np.linspace(0.1, 2 * math.pi - 0.1, 8).tolist()
        assert not any(_classify_points(ellipse, ts, circumscribed=False))

    def test_star_tip_circumscribed_is_contact(self, star):
        """星形先端（外側凸）は外包構築で接点"""
        assert _classify_points(star, [0.0], circumscribed=True)[0] is True

    def test_star_valley_circumscribed_is_vertex(self, star):
        """星形の谷（内側凸）は外包構築で頂点"""
        assert _classify_points(star, [math.pi / 5], circumscribed=True)[0] is False


# -----------------------------------------------------------------------
# _find_zero_curvature
# -----------------------------------------------------------------------

class TestFindZeroCurvature:
    # 曲率零点で許容する |κ| の上限
    _KAPPA_TOL = 1e-6

    def test_ten_zero_crossings(self, star):
        """StarCurve(r=5, a=0.8, n=5) の連続する極値間 10 区間で零点が 1 個ずつ得られる"""
        pi = math.pi
        extrema = sorted([
            0.0, 2*pi/5, 4*pi/5, 6*pi/5, 8*pi/5,  # 極大
            pi/5, 3*pi/5, pi, 7*pi/5, 9*pi/5,       # 極小
        ])
        zeros = [
            _find_zero_curvature(star, extrema[i], extrema[(i + 1) % len(extrema)])
            for i in range(len(extrema))
        ]
        assert len(zeros) == 10
        for t in zeros:
            assert abs(star.signed_curvature(t)) <= self._KAPPA_TOL


# -----------------------------------------------------------------------
# _find_curvature_extrema
# -----------------------------------------------------------------------

def _circ_dist(a: float, b: float, period: float) -> float:
    """周期的距離"""
    d = abs(a - b) % period
    return min(d, period - d)


class TestFindCurvatureExtrema:
    def test_star_extrema_count(self, star):
        """StarCurve(r=5, a=0.8, n=5) の極値数: 極大5・極小5"""
        maxima, minima = _find_curvature_extrema(star)
        assert len(maxima) == 5
        assert len(minima) == 5

    def test_star_maxima_positions(self, star):
        """極大点は t = 0, 2π/5, 4π/5, 6π/5, 8π/5 の近傍"""
        maxima, _ = _find_curvature_extrema(star)
        period = 2 * math.pi
        expected = [0.0, 2*math.pi/5, 4*math.pi/5, 6*math.pi/5, 8*math.pi/5]
        for e in expected:
            assert any(_circ_dist(m, e, period) < 1e-2 for m in maxima)

    def test_star_minima_positions(self, star):
        """極小点は t = π/5, 3π/5, π, 7π/5, 9π/5 の近傍"""
        _, minima = _find_curvature_extrema(star)
        period = 2 * math.pi
        expected = [math.pi/5, 3*math.pi/5, math.pi, 7*math.pi/5, 9*math.pi/5]
        for e in expected:
            assert any(_circ_dist(m, e, period) < 1e-2 for m in minima)

    def test_star_maxima_are_local_maxima(self, star):
        """各極大点で両側の近傍より曲率が大きい"""
        maxima, _ = _find_curvature_extrema(star)
        for t in maxima:
            k = star.signed_curvature(t)
            assert k > star.signed_curvature(t - 0.01)
            assert k > star.signed_curvature(t + 0.01)

    def test_star_minima_are_local_minima(self, star):
        """各極小点で両側の近傍より曲率が小さい"""
        _, minima = _find_curvature_extrema(star)
        for t in minima:
            k = star.signed_curvature(t)
            assert k < star.signed_curvature(t - 0.01)
            assert k < star.signed_curvature(t + 0.01)
