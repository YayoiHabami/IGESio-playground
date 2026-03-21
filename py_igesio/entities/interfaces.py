import abc

import numpy as np


class CurveBase2D(abc.ABC):
    """任意の2次元パラメトリック曲線の基底クラス。"""

    @abc.abstractmethod
    def get_name(self) -> str:
        """曲線の名前（テスト用）"""

    @abc.abstractmethod
    def is_closed(self) -> bool:
        """閉曲線であるか"""

    @abc.abstractmethod
    def has_self_intersection(self) -> bool:
        """自己交差を持つか"""

    @abc.abstractmethod
    def is_clockwise(self) -> bool:
        """時計回りか"""

    @abc.abstractmethod
    def get_range(self) -> tuple[float, float]:
        """tの定義域 (t_min, t_max)"""

    @abc.abstractmethod
    def evaluate(self, t: float) -> np.ndarray:
        """C(t) -> (x, y)"""

    @abc.abstractmethod
    def derivative(self, t: float, order: int) -> np.ndarray:
        """C'(t) または C''(t) -> (dx, dy) または (ddx, ddy)

        角点 (corner point) では呼び出してはならない。
        角点かどうかは get_corner_params() で事前に確認すること。
        """

    # ------------------------------------------------------------------
    # 角点 (corner point) 対応
    # ------------------------------------------------------------------

    def get_corner_params(self) -> list[float]:
        """角点のパラメータ値のリストを返す。

        角点を持たない曲線ではデフォルトの空リストがそのまま使われる。
        サブクラスで角点がある場合にオーバーライドする。
        """
        return []

    def get_linear_segments(self) -> list[tuple[float, float]]:
        """直線部のパラメータ区間リストを返す。

        直線部を持たない曲線ではデフォルトの空リストがそのまま使われる。
        サブクラスで直線部がある場合にオーバーライドする。
        例: [(u_s1, u_e1), (u_s2, u_e2), ...]
        各区間は u_s < u_e かつ get_range() の範囲内であること。
        """
        return []

    def is_corner(self, t: float, eps: float = 1e-9) -> bool:
        """パラメータ値 t が角点かどうかを判定する。"""
        for tc in self.get_corner_params():
            if abs(t - tc) < eps:
                return True
        return False

    def left_tangent(self, t: float) -> np.ndarray:
        """角点における左側単位接線ベクトル T⁻(t)。

        デフォルト実装は t の左側近傍から接線を近似する。
        サブクラスで解析的に左側導関数が既知の場合はオーバーライド可能。
        """
        h = 1e-7
        t_min, _ = self.get_range()
        t_eval = max(t - h, t_min)
        d = self.derivative(t_eval, 1)
        return d / np.linalg.norm(d)

    def right_tangent(self, t: float) -> np.ndarray:
        """角点における右側単位接線ベクトル T⁺(t)。

        デフォルト実装は t の右側近傍から接線を近似する。
        サブクラスで解析的に右側導関数が既知の場合はオーバーライド可能。
        """
        h = 1e-7
        _, t_max = self.get_range()
        t_eval = min(t + h, t_max)
        d = self.derivative(t_eval, 1)
        return d / np.linalg.norm(d)

    def corner_exterior_angle(self, t: float) -> float:
        """角点における外角 α を返す。

        T⁻ から T⁺ への符号付き回転角（ラジアン）。
        α > 0: 反時計回りの折れ, α < 0: 時計回りの折れ。
        """
        Tm = self.left_tangent(t)
        Tp = self.right_tangent(t)
        cross = Tm[0] * Tp[1] - Tm[1] * Tp[0]
        dot = Tm[0] * Tp[0] + Tm[1] * Tp[1]
        return np.arctan2(cross, dot)

    # ------------------------------------------------------------------
    # 符号付き曲率（角点対応）
    # ------------------------------------------------------------------

    def signed_curvature(self, t: float) -> float:
        """符号付き曲率 κ_s(t) を計算する。

        滑らかな点では通常の定義式を用い、
        角点では外角の符号に基づき ±∞ を返す。
        """
        if self.is_corner(t):
            alpha = self.corner_exterior_angle(t)
            if alpha > 0:
                return float('inf')
            elif alpha < 0:
                return float('-inf')
            else:
                return 0.0

        d1 = self.derivative(t, 1)
        d2 = self.derivative(t, 2)
        xp, yp = d1
        xpp, ypp = d2
        denom = (xp**2 + yp**2) ** 1.5
        if abs(denom) < 1e-14:
            return 0.0
        return (xp * ypp - yp * xpp) / denom

    # ------------------------------------------------------------------
    # 接線・法線（角点対応）
    # ------------------------------------------------------------------

    def tangent(self, t: float) -> np.ndarray:
        """単位接線ベクトル T(t)。

        角点では左側接線と右側接線の二等分方向を返す。
        """
        if self.is_corner(t):
            return self._bisect_tangent(t)

        d = self.derivative(t, 1)
        return d / np.linalg.norm(d)

    def _bisect_tangent(self, t: float) -> np.ndarray:
        """角点における二等分接線ベクトルを返す。"""
        Tm = self.left_tangent(t)
        Tp = self.right_tangent(t)
        bisect = Tm + Tp
        norm = np.linalg.norm(bisect)
        if norm < 1e-12:
            # T⁻ と T⁺ が正反対 → T⁻ の90度回転
            return np.array([-Tm[1], Tm[0]])
        return bisect / norm

    def normal(self, t: float) -> np.ndarray:
        """単位法線ベクトル N(t)"""
        if self.is_corner(t):
            T = self._bisect_tangent(t)
            return np.array([-T[1], T[0]])

        d1 = self.derivative(t, 1)
        d2 = self.derivative(t, 2)
        speed = np.linalg.norm(d1)
        # dT/dt = (d2 - (d2·T)T) / speed
        T = d1 / speed
        dT = (d2 - np.dot(d2, T) * T) / speed
        n = np.linalg.norm(dT)
        if n < 1e-12:
            # 曲率ゼロ付近では接線の90度回転を返す
            return np.array([-T[1], T[0]])
        return dT / n

    def get_length(self) -> float:
        """曲線の長さ（数値積分）"""
        from scipy.integrate import quad
        t0, t1 = self.get_range()
        def integrand(t):
            d = self.derivative(t, 1)
            return np.linalg.norm(d)
        length, _ = quad(integrand, t0, t1, limit=200)
        return length
