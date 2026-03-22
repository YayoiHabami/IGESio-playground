"""CurveBase2D (entities/interfaces.py) の単体テスト"""
import math

import pytest
import numpy as np

from py_igesio.entities.curves import EllipseCurve, PolylineCurve2D


# -----------------------------------------------------------------------
# フィクスチャ
# -----------------------------------------------------------------------

@pytest.fixture
def ellipse():
    """a=2, b=1 の楕円曲線（角点なし・CCW）"""
    return EllipseCurve(a=2.0, b=1.0)


@pytest.fixture
def unit_square():
    """単位正方形ポリライン（CCW、外角 +π/2）"""
    return PolylineCurve2D([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def cw_triangle():
    """直角三角形ポリライン（CW、外角 < 0）"""
    return PolylineCurve2D([(0, 0), (0, 1), (1, 0)])


# -----------------------------------------------------------------------
# get_corner_params / get_linear_segments のデフォルト実装
# -----------------------------------------------------------------------

class TestDefaultCornerMethods:
    def test_smooth_curve_no_corners(self, ellipse):
        assert ellipse.get_corner_params() == []

    def test_smooth_curve_no_linear_segments(self, ellipse):
        assert ellipse.get_linear_segments() == []


# -----------------------------------------------------------------------
# is_corner
# -----------------------------------------------------------------------

class TestIsCorner:
    def test_smooth_curve_never_corner(self, ellipse):
        for t in [0.0, 0.5, math.pi, 2 * math.pi - 0.1]:
            assert ellipse.is_corner(t) is False

    def test_polyline_integer_params_are_corners(self, unit_square):
        for t in [0.0, 1.0, 2.0, 3.0]:
            assert unit_square.is_corner(t) is True

    def test_polyline_non_integer_params_are_not_corners(self, unit_square):
        for t in [0.5, 1.5, 2.5, 3.5]:
            assert unit_square.is_corner(t) is False

    def test_is_corner_within_eps(self, unit_square):
        # デフォルト eps=1e-9 内なら角点と判定
        assert unit_square.is_corner(1.0 + 1e-10) is True

    def test_is_corner_outside_eps(self, unit_square):
        # eps より外なら非角点
        assert unit_square.is_corner(1.0 + 1e-8) is False


# -----------------------------------------------------------------------
# left_tangent / right_tangent のデフォルト実装（EllipseCurve）
# -----------------------------------------------------------------------

class TestDefaultTangents:
    """EllipseCurve は left/right_tangent をオーバーライドしないため、
    デフォルトの数値近似実装が使われる。"""

    def test_left_tangent_is_unit_vector(self, ellipse):
        for t in [0.3, 1.0, 2.5]:
            v = ellipse.left_tangent(t)
            assert pytest.approx(np.linalg.norm(v), abs=1e-6) == 1.0

    def test_right_tangent_is_unit_vector(self, ellipse):
        for t in [0.3, 1.0, 2.5]:
            v = ellipse.right_tangent(t)
            assert pytest.approx(np.linalg.norm(v), abs=1e-6) == 1.0

    def test_left_right_tangent_agree_on_smooth_curve(self, ellipse):
        """滑らかな曲線では左右の接線が一致する"""
        for t in [0.5, 1.2, 2.0]:
            left = ellipse.left_tangent(t)
            right = ellipse.right_tangent(t)
            np.testing.assert_allclose(left, right, atol=1e-5)


# -----------------------------------------------------------------------
# corner_exterior_angle
# -----------------------------------------------------------------------

class TestCornerExteriorAngle:
    def test_ccw_square_corner_angle_is_pi_over_2(self, unit_square):
        """CCW 正方形の各角点の外角は +π/2"""
        for t in [0.0, 1.0, 2.0, 3.0]:
            angle = unit_square.corner_exterior_angle(t)
            assert pytest.approx(angle, abs=1e-10) == math.pi / 2

    def test_cw_triangle_corner_angle_is_negative(self, cw_triangle):
        """CW 折れ線の角点の外角は負"""
        for t in [0.0, 1.0, 2.0]:
            angle = cw_triangle.corner_exterior_angle(t)
            assert angle < 0


# -----------------------------------------------------------------------
# signed_curvature
# -----------------------------------------------------------------------

class TestSignedCurvature:
    def test_ellipse_curvature_at_major_axis_end(self, ellipse):
        """t=0（長軸端）での曲率: κ = ab / b^3 = a / b^2 = 2"""
        kappa = ellipse.signed_curvature(0.0)
        assert pytest.approx(kappa, rel=1e-6) == 2.0

    def test_ellipse_curvature_at_minor_axis_end(self, ellipse):
        """t=π/2（短軸端）での曲率: κ = ab / a^3 = b / a^2 = 0.25"""
        kappa = ellipse.signed_curvature(math.pi / 2)
        assert pytest.approx(kappa, rel=1e-6) == 0.25

    def test_ellipse_curvature_positive_throughout_ccw(self, ellipse):
        """CCW 曲線の曲率は全域で正"""
        for t in np.linspace(0.1, 2 * math.pi - 0.1, 20):
            assert ellipse.signed_curvature(t) > 0

    def test_corner_positive_exterior_angle_gives_pos_inf(self, unit_square):
        """外角 > 0 の角点は +∞"""
        kappa = unit_square.signed_curvature(1.0)
        assert kappa == float('inf')

    def test_corner_negative_exterior_angle_gives_neg_inf(self, cw_triangle):
        """外角 < 0 の角点は -∞"""
        kappa = cw_triangle.signed_curvature(0.0)
        assert kappa == float('-inf')


# -----------------------------------------------------------------------
# tangent
# -----------------------------------------------------------------------

class TestTangent:
    def test_tangent_is_unit_vector(self, ellipse):
        for t in [0.1, 0.5, 1.0, 2.0]:
            T = ellipse.tangent(t)
            assert pytest.approx(np.linalg.norm(T), abs=1e-10) == 1.0

    def test_tangent_direction_at_t0(self, ellipse):
        """t=0: derivative=(0, b), 単位化 → (0, 1)"""
        T = ellipse.tangent(0.0)
        np.testing.assert_allclose(T, [0.0, 1.0], atol=1e-10)

    def test_tangent_at_corner_is_bisect(self, unit_square):
        """t=1: T⁻=(1,0)、T⁺=(0,1) の二等分 → (1,1)/√2"""
        T = unit_square.tangent(1.0)
        expected = np.array([1.0, 1.0]) / math.sqrt(2)
        np.testing.assert_allclose(T, expected, atol=1e-10)

    def test_tangent_at_corner_is_unit_vector(self, unit_square):
        T = unit_square.tangent(1.0)
        assert pytest.approx(np.linalg.norm(T), abs=1e-10) == 1.0


# -----------------------------------------------------------------------
# normal
# -----------------------------------------------------------------------

class TestNormal:
    def test_normal_is_unit_vector(self, ellipse):
        for t in [0.1, 0.5, 1.0, 2.0]:
            N = ellipse.normal(t)
            assert pytest.approx(np.linalg.norm(N), abs=1e-10) == 1.0

    def test_normal_perpendicular_to_tangent(self, ellipse):
        for t in [0.1, 0.5, 1.0, 2.0]:
            T = ellipse.tangent(t)
            N = ellipse.normal(t)
            assert pytest.approx(float(np.dot(T, N)), abs=1e-8) == 0.0

    def test_normal_at_corner_is_unit_vector(self, unit_square):
        N = unit_square.normal(1.0)
        assert pytest.approx(np.linalg.norm(N), abs=1e-10) == 1.0

    def test_normal_at_corner_perpendicular_to_tangent(self, unit_square):
        T = unit_square.tangent(1.0)
        N = unit_square.normal(1.0)
        assert pytest.approx(float(np.dot(T, N)), abs=1e-10) == 0.0


# -----------------------------------------------------------------------
# get_length
# -----------------------------------------------------------------------

class TestGetLength:
    def test_circle_circumference(self):
        """a=b=1 の楕円（単位円）の長さは 2π"""
        circle = EllipseCurve(a=1.0, b=1.0)
        length = circle.get_length()
        assert pytest.approx(length, rel=1e-5) == 2 * math.pi

    def test_square_perimeter(self, unit_square):
        """単位正方形の周長は 4"""
        length = unit_square.get_length()
        assert pytest.approx(length, rel=1e-5) == 4.0
