"""_alg_polygonal_approximation.py の単体テスト"""
import math

import pytest
import numpy as np

from py_igesio.entities.interfaces import CurveBase2D
from py_igesio.entities.curves import EllipseCurve, PolylineCurve2D
from py_igesio.entities.curves._alg_polygonal_approximation import (
    _is_in_linear_segment,
    _point_to_line_distance,
    _subdivide,
)


class _MockLineCurve(CurveBase2D):
    """テスト用: 直線 y=0 の曲線（直線部セグメントを報告しない）"""
    def get_name(self): return "MockLine"
    def is_closed(self): return False
    def has_self_intersection(self): return False
    def is_clockwise(self): return False
    def get_range(self): return (0.0, 1.0)
    def evaluate(self, t): return np.array([t, 0.0])
    def derivative(self, t, order): return np.array([1.0, 0.0])


# -----------------------------------------------------------------------
# _is_in_linear_segment
# -----------------------------------------------------------------------

class TestIsInLinearSegment:
    def test_empty_segments_returns_false(self):
        assert _is_in_linear_segment(0.0, 1.0, []) is False

    def test_fully_contained_returns_true(self):
        assert _is_in_linear_segment(0.2, 0.8, [(0.0, 1.0)]) is True

    def test_exact_match_returns_true(self):
        assert _is_in_linear_segment(0.0, 1.0, [(0.0, 1.0)]) is True

    def test_left_endpoint_outside_returns_false(self):
        """u_a がセグメント開始より前"""
        assert _is_in_linear_segment(-0.1, 0.8, [(0.0, 1.0)]) is False

    def test_right_endpoint_outside_returns_false(self):
        """u_b がセグメント終端より後"""
        assert _is_in_linear_segment(0.2, 1.1, [(0.0, 1.0)]) is False

    def test_matches_second_segment(self):
        """複数セグメントの 2 番目に一致"""
        assert _is_in_linear_segment(2.0, 3.0, [(0.0, 1.0), (2.0, 3.0)]) is True


# -----------------------------------------------------------------------
# _point_to_line_distance
# -----------------------------------------------------------------------

class TestPointToLineDistance:
    def test_point_on_line_is_zero(self):
        """x 軸上の点は距離 0"""
        d = _point_to_line_distance(
            np.array([0.5, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        )
        assert pytest.approx(d, abs=1e-12) == 0.0

    def test_point_off_axis_returns_correct_distance(self):
        """(0.5, 3) から x 軸（a=(0,0), b=(1,0)）への距離は 3"""
        d = _point_to_line_distance(
            np.array([0.5, 3.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
        )
        assert pytest.approx(d, abs=1e-12) == 3.0

    def test_diagonal_line_known_value(self):
        """y=x（a=(0,0), b=(1,1)）に対する点 (1,0) の距離は 1/√2"""
        d = _point_to_line_distance(
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        )
        assert pytest.approx(d, abs=1e-12) == 1.0 / math.sqrt(2)

    def test_degenerate_ab_returns_point_distance(self):
        """a == b の場合は点 p から a への距離"""
        d = _point_to_line_distance(
            np.array([3.0, 4.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        assert pytest.approx(d, abs=1e-12) == 5.0


# -----------------------------------------------------------------------
# _subdivide
# -----------------------------------------------------------------------

class TestSubdivide:
    def test_linear_segment_returns_endpoints_only(self):
        """直線部セグメント上の区間は両端のみ返る"""
        curve = PolylineCurve2D([(0, 0), (1, 0), (1, 1)])
        result = _subdivide(curve, 0.0, 1.0, eps=1e-6, max_depth=100)
        assert result == [0.0, 1.0]

    def test_straight_curve_no_linear_segment_returns_endpoints_only(self):
        """中点距離 0 の直線（直線部セグメント未報告）は eps>0 で両端のみ"""
        curve = _MockLineCurve()
        result = _subdivide(curve, 0.0, 1.0, eps=1e-6, max_depth=100)
        assert result == [0.0, 1.0]

    def test_endpoints_always_present(self):
        """u_a が先頭、u_b が末尾"""
        curve = EllipseCurve(a=1.0, b=1.0)
        result = _subdivide(curve, 0.0, math.pi, eps=1e-3, max_depth=100)
        assert result[0] == 0.0
        assert result[-1] == pytest.approx(math.pi, abs=1e-12)

    def test_result_is_sorted(self):
        """返値は昇順"""
        curve = EllipseCurve(a=1.0, b=1.0)
        result = _subdivide(curve, 0.0, math.pi, eps=1e-3, max_depth=100)
        assert result == sorted(result)

    def test_curved_section_produces_more_than_two_points(self):
        """曲率のある弧は eps が小さいと 2 点より多くなる"""
        curve = EllipseCurve(a=1.0, b=1.0)
        result = _subdivide(curve, 0.0, math.pi, eps=0.01, max_depth=100)
        assert len(result) > 2

    def test_max_depth_zero_yields_three_points(self):
        """max_depth=0: depth=0 で即座に分割停止 → [u_a, u_mid, u_b] の 3 点"""
        curve = EllipseCurve(a=1.0, b=1.0)
        u_a, u_b = 0.0, math.pi
        result = _subdivide(curve, u_a, u_b, eps=0.0, max_depth=0)
        assert len(result) == 3
        assert result[0] == u_a
        assert result[1] == pytest.approx((u_a + u_b) / 2, abs=1e-12)
        assert result[-1] == pytest.approx(u_b, abs=1e-12)

    def test_smaller_eps_yields_more_points(self):
        """eps を小さくすると点数が増加する"""
        curve = EllipseCurve(a=1.0, b=1.0)
        r_coarse = _subdivide(curve, 0.0, math.pi, eps=0.1, max_depth=100)
        r_fine   = _subdivide(curve, 0.0, math.pi, eps=0.001, max_depth=100)
        assert len(r_fine) > len(r_coarse)
