"""
曲線エンティティの実装

IGESioの実装とは大きく異なる (EllipseCurveやStarCurveはIGESioには存在しない) が、
テストやデモを目的として、簡単な曲線クラスをいくつか実装する。

Classes
-------
- EllipseCurve: 楕円曲線クラス
- StarCurve: 星形曲線クラス
- NURBSCurve: geomdl を用いたNURBS曲線クラス
- PolylineCurve2D: 折れ線による閉曲線クラス
"""
from typing import Sequence, Optional

from geomdl import NURBS
import numpy as np

from py_igesio.entities.interfaces import CurveBase2D



class EllipseCurve(CurveBase2D):
    """楕円 x=a*cos(t), y=b*sin(t), t in [0, 2π)"""

    def __init__(self, a: float = 2.0, b: float = 1.0):
        self.a = a
        self.b = b

    def get_name(self) -> str:
        return f"Ellipse(a={self.a}, b={self.b})"

    def is_closed(self) -> bool:
        return True

    def has_self_intersection(self) -> bool:
        return False

    def is_clockwise(self) -> bool:
        # x'y'' - y'x'' = -a*sin(t)*(-b*sin(t)) - b*cos(t)*(-a*cos(t))
        #                = ab*sin²(t) + ab*cos²(t) = ab > 0  → CCW
        return False

    def get_range(self) -> tuple[float, float]:
        return (0.0, 2 * np.pi)

    def evaluate(self, t: float) -> np.ndarray:
        return np.array([self.a * np.cos(t), self.b * np.sin(t)])

    def derivative(self, t: float, order: int) -> np.ndarray:
        if order == 1:
            return np.array([-self.a * np.sin(t), self.b * np.cos(t)])
        if order == 2:
            return np.array([-self.a * np.cos(t), -self.b * np.sin(t)])
        raise ValueError(f"order must be 1 or 2, got {order}")

class StarCurve(CurveBase2D):
    """
    星形曲線 r(t) = R + A*cos(n*t)
    x = r(t)*cos(t), y = r(t)*sin(t)
    デフォルトは5角星。
    """

    def __init__(self, r: float = 2.0, a: float = 1.0, n: int = 5):
        self.r = r
        self.a = a
        self.n = n

    def get_name(self) -> str:
        return f"Star(r={self.r}, a={self.a}, n={self.n})"

    def is_closed(self) -> bool:
        return True

    def has_self_intersection(self) -> bool:
        # r > a なら自己交差なし
        return self.r <= self.a

    def is_clockwise(self) -> bool:
        # 数値積分で向きを判定（面積の符号）
        t0, t1 = self.get_range()
        ts = np.linspace(t0, t1, 1000)
        pts = np.array([self.evaluate(t) for t in ts])
        # 台形法で符号付き面積
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        return area < 0

    def get_range(self) -> tuple[float, float]:
        return (0.0, 2 * np.pi)

    def _r(self, t):
        return self.r + self.a * np.cos(self.n * t)

    def _rp(self, t):
        return -self.a * self.n * np.sin(self.n * t)

    def _rpp(self, t):
        return -self.a * self.n**2 * np.cos(self.n * t)

    def evaluate(self, t: float) -> np.ndarray:
        r = self._r(t)
        return np.array([r * np.cos(t), r * np.sin(t)])

    def derivative(self, t: float, order: int) -> np.ndarray:
        r = self._r(t)
        rp = self._rp(t)
        if order == 1:
            xp = rp * np.cos(t) - r * np.sin(t)
            yp = rp * np.sin(t) + r * np.cos(t)
            return np.array([xp, yp])
        if order == 2:
            rpp = self._rpp(t)
            xpp = rpp * np.cos(t) - 2 * rp * np.sin(t) - r * np.cos(t)
            ypp = rpp * np.sin(t) + 2 * rp * np.cos(t) - r * np.sin(t)
            return np.array([xpp, ypp])
        raise ValueError(f"order must be 1 or 2, got {order}")

class NURBSCurve(CurveBase2D):
    """
    geomdl を用いた NURBS曲線。

    制御点・重み・ノットベクトルを直接受け取り、
    CurveBase2D と同じインターフェースで評価できる。

    パラメータ空間は [0, 1] に正規化される (geomdl の規約)。
    """

    def __init__(
        self,
        control_points: list[tuple[float, float, float]],
        weights: list[float],
        knot_vector: list[float],
        degree: int = 3,
    ):
        """
        Parameters
        ----------
        control_points : list of (x, y, z)
            同次座標ではなく通常の 3D 座標で渡す（z=0 で平面曲線）。
        weights       : list of float
            各制御点の重み（長さは control_points と同じ）。
        knot_vector   : list of float
            正規化済みノットベクトル（0〜1 の範囲）。
            長さは len(control_points) + degree + 1 でなければならない。
        degree        : int
            NURBS 曲線の次数（デフォルト 3）。
        """
        if len(control_points) != len(weights):
            raise ValueError(
                "control_points と weights の長さが一致しません: "
                f"{len(control_points)} vs {len(weights)}"
            )
        expected_knot_len = len(control_points) + degree + 1
        if len(knot_vector) != expected_knot_len:
            raise ValueError(
                f"knot_vector の長さが不正です。期待値={expected_knot_len}, "
                f"実際={len(knot_vector)}"
            )

        self._ctrl_pts = control_points
        self._weights = weights
        self._knot_vector = knot_vector
        self._degree = degree
        self._is_clockwise: Optional[bool] = None

        # geomdl の NURBS.Curve を構築
        self._curve = self._build_curve()

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _build_curve(self) -> NURBS.Curve:
        """geomdl の NURBS.Curve オブジェクトを構築して返す。"""
        curve = NURBS.Curve()
        curve.degree = self._degree

        # geomdl は重み付き同次座標 [x*w, y*w, z*w, w] で制御点を受け取る
        weighted_ctrl_pts = [
            [x * w, y * w, z * w, w]
            for (x, y, z), w in zip(self._ctrl_pts, self._weights)
        ]
        curve.ctrlptsw = weighted_ctrl_pts
        curve.knotvector = self._knot_vector

        return curve

    def _t_to_geomdl(self, t: float) -> float:
        """
        外部パラメータ t ∈ [0, 1] をそのまま geomdl パラメータへ変換する。
        （geomdl は既に [0, 1] を使うのでパススルー）
        """
        return float(np.clip(t, 0.0, 1.0))

    def _evaluate_raw(self, t: float) -> np.ndarray:
        """geomdl で点を評価し (x, y) の ndarray を返す。"""
        u = self._t_to_geomdl(t)
        pt = self._curve.evaluate_single(u)   # [x, y, z] を返す
        return np.array(pt[:2], dtype=float)  # z を捨てて 2D に

    def _derivative_raw(self, t: float, order: int) -> list[np.ndarray]:
        """
        geomdl の derivatives() を用いて 0〜order 次の導関数を返す。

        Returns
        -------
        list of ndarray, shape (order+1,)
            [C(t), C'(t), C''(t), ...]  各要素は (x, y) の 2D ベクトル
        """
        u = self._t_to_geomdl(t)
        # derivatives() は [[C(u)], [C'(u)], ...] の形式で返す
        deriv_s = self._curve.derivatives(u, order=order)
        return [np.array(d[:2], dtype=float) for d in deriv_s]

    # ------------------------------------------------------------------
    # CurveBase2D インターフェース実装
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return (
            f"NURBSCurve("
            f"degree={self._degree}, "
            f"n_ctrl={len(self._ctrl_pts)})"
        )

    def is_closed(self) -> bool:
        """
        最初と最後の制御点が一致するかどうかで閉曲線を判定する。
        """
        first = np.array(self._ctrl_pts[0])
        last = np.array(self._ctrl_pts[-1])
        return bool(np.allclose(first, last))

    def has_self_intersection(self) -> bool:
        """
        数値的サンプリングにより自己交差の有無を簡易判定する。
        閉曲線の場合、端点付近の重複区間は除外して判定する。
        """
        n_sample = 50
        ts = np.linspace(0.0, 1.0, n_sample, endpoint=False)
        pts = np.array([self._evaluate_raw(t) for t in ts])

        # O(n^2) で全ペアを比較（小規模サンプルなので許容範囲）
        len_pts = len(pts)
        for i in range(len_pts):
            for j in range(i + 2, len_pts):
                # 閉曲線の場合、始点と終点近傍の組み合わせはスキップ
                if self.is_closed() and j == len_pts - 1 and i == 0:
                    continue
                if np.linalg.norm(pts[i] - pts[j]) < 1e-3:
                    return True
        return False

    def is_clockwise(self) -> bool:
        """
        Shoelace formula（台形則）で符号付き面積を計算し向きを判定する。
        """
        if self._is_clockwise is not None:
            return self._is_clockwise

        # Shoelace formula
        ts = np.linspace(0.0, 1.0, 100)
        pts = np.array([self._evaluate_raw(t) for t in ts])
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        return area < 0

    def get_range(self) -> tuple[float, float]:
        """パラメータ範囲を返す（geomdl 規約に合わせ [0, 1]）。"""
        return (0.0, 1.0)

    def evaluate(self, t: float) -> np.ndarray:
        """
        パラメータ t ∈ [0, 1] における曲線上の点 (x, y) を返す。

        Parameters
        ----------
        t : float
            曲線パラメータ（0〜1）。

        Returns
        -------
        np.ndarray, shape (2,)
        """
        return self._evaluate_raw(t)

    def derivative(self, t: float, order: int) -> np.ndarray:
        """
        パラメータ t における order 次導関数ベクトルを返す。

        Parameters
        ----------
        t     : float  曲線パラメータ（0〜1）。
        order : int    微分次数（1 または 2）。

        Returns
        -------
        np.ndarray, shape (2,)
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        deriv_s = self._derivative_raw(t, order)
        return deriv_s[order]

class PolylineCurve2D(CurveBase2D):
    """
    折れ線による閉曲線クラス。

    頂点列 vertices を受け取り、最後の頂点から最初の頂点へ自動的に閉じる。
    パラメータ t ∈ [0, N]（N = 頂点数）で各辺を等速で走る。
    各頂点が角点 (corner point) として登録される。

    Parameters
    ----------
    vertices : array-like, shape (N, 2)
        折れ線の頂点座標列。最低3点必要。
        最初と最後の頂点が同じ場合は自動的に重複を除去する。
    """

    def __init__(self, vertices: Sequence):
        pts = np.asarray(vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("vertices must be shape (N, 2)")

        # 末尾が先頭と重複している場合は除去して閉じる
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]

        if len(pts) < 3:
            raise ValueError("At least 3 distinct vertices are required")

        # 閉じるために先頭を末尾に追加（内部保持用）
        self._verts = pts                          # shape (N, 2)
        self._n = len(pts)                         # セグメント数 = 頂点数

        # 各セグメントのベクトルと長さを事前計算
        closed = np.vstack([pts, pts[0]])          # shape (N+1, 2)
        self._segments = np.diff(closed, axis=0)   # shape (N, 2)  各辺のベクトル
        self._lengths = np.linalg.norm(self._segments, axis=1)  # shape (N,)

        if np.any(self._lengths == 0):
            raise ValueError("Duplicate consecutive vertices are not allowed")

        # 単位接線ベクトル（セグメントごと）
        self._tangents = self._segments / self._lengths[:, np.newaxis]  # shape (N, 2)

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    def _segment_index(self, t: float) -> tuple[int, float]:
        """t からセグメントインデックス i と局所パラメータ s ∈ [0,1) を返す。"""
        t_min, t_max = self.get_range()
        # 範囲外をクランプ（浮動小数誤差対策）
        t = float(np.clip(t, t_min, t_max))
        if t == t_max:
            # 閉じた終端は最終セグメントの末端
            return self._n - 1, 1.0
        i = int(t)
        i = min(i, self._n - 1)
        s = t - i
        return i, s

    # ------------------------------------------------------------------
    # CurveBase2D 抽象メソッドの実装
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return f"PolylineCurve2D(n_vertices={self._n})"

    def is_closed(self) -> bool:
        return True

    def has_self_intersection(self) -> bool:
        """全セグメントペアの交差判定（O(N²)）。"""

        def _cross2d(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def _segments_intersect(p, r, q, s):
            """線分 p+t*r と q+u*s が 0<t<1, 0<u<1 で交差するか。"""
            denom = _cross2d(r, s)
            if abs(denom) < 1e-12:
                return False  # 平行
            diff = q - p
            t_ = _cross2d(diff, s) / denom
            u_ = _cross2d(diff, r) / denom
            eps = 1e-9
            return eps < t_ < 1 - eps and eps < u_ < 1 - eps

        n = self._n
        closed = np.vstack([self._verts, self._verts[0]])
        for i in range(n):
            p = closed[i]
            r = self._segments[i]
            for j in range(i + 2, n):
                # 隣接セグメント（共有頂点あり）はスキップ
                if i == 0 and j == n - 1:
                    continue
                q = closed[j]
                s = self._segments[j]
                if _segments_intersect(p, r, q, s):
                    return True
        return False

    def is_clockwise(self) -> bool:
        """符号付き面積（Shoelace formula）で向きを判定。"""
        x = self._verts[:, 0]
        y = self._verts[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        return area < 0

    def get_range(self) -> tuple[float, float]:
        """t ∈ [0, N]（N = セグメント数）"""
        return (0.0, float(self._n))

    def evaluate(self, t: float) -> np.ndarray:
        """折れ線上の点 C(t)。"""
        i, s = self._segment_index(t)
        return self._verts[i] + s * self._segments[i]

    def derivative(self, t: float, order: int) -> np.ndarray:
        """
        折れ線の導関数。

        1階：セグメントの速度ベクトル（長さ = セグメント長）。
        2階：0ベクトル（直線セグメントの加速度は恒等的に 0）。
        角点では呼び出してはならない。
        """
        if order == 1:
            i, _ = self._segment_index(t)
            return self._segments[i].copy()
        if order == 2:
            return np.zeros(2)
        raise ValueError(f"order must be 1 or 2, got {order}")

    # ------------------------------------------------------------------
    # 角点サポート
    # ------------------------------------------------------------------

    def get_corner_params(self) -> list[float]:
        """各頂点のパラメータ値（整数値 0, 1, ..., N-1）を返す。"""
        return [float(i) for i in range(self._n)]

    def get_linear_segments(self) -> list[tuple[float, float]]:
        """直線部のパラメータ区間リストを返す。

        直線部を持たない曲線ではデフォルトの空リストがそのまま使われる。
        サブクラスで直線部がある場合にオーバーライドする。
        例: [(u_s1, u_e1), (u_s2, u_e2), ...]
        各区間は u_s < u_e かつ get_range() の範囲内であること。
        """
        return [(float(i), float(i+1)) for i in range(self._n)]

    def left_tangent(self, t: float) -> np.ndarray:
        """角点における左側（入射）単位接線ベクトル。"""
        i = int(round(t)) % self._n
        prev_i = (i - 1) % self._n
        return self._tangents[prev_i].copy()

    def right_tangent(self, t: float) -> np.ndarray:
        """角点における右側（出射）単位接線ベクトル。"""
        i = int(round(t)) % self._n
        return self._tangents[i].copy()
