"""
外包/内包多角形 (閉曲線C(t)を完全に内包する/完全に内包される多角形) を構築するアルゴリズム。

NOTE: C++移植時は`src/entities/curves/algorithms/extremal_polygon.h(cpp)`に実装する (APIに含めない) 。
"""
import time as _time  # デバッグ用

import numpy as np
from scipy import optimize as _scipy_optimize

from py_igesio.numerics.polygons import PolygonData
from ..interfaces import CurveBase2D



# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------

def _line_intersect(p1: np.ndarray, d1: np.ndarray,
                    p2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    直線 p1 + s*d1 と p2 + t*d2 の交点を返す。
    平行な場合は中点を返す（縮退ケース）。
    """
    # p1 + s*d1 = p2 + t*d2
    # [d1, -d2] [s; t] = p2 - p1
    mat_a = np.column_stack([d1, -d2])
    b = p2 - p1
    if abs(np.linalg.det(mat_a)) < 1e-12:
        # 平行 or 一致 → 中点
        return (p1 + p2) / 2.0
    st = np.linalg.solve(mat_a, b)
    return p1 + st[0] * d1


def _find_zero_curvature(curve: CurveBase2D, ta: float, tb: float,
                         eps: float = 1e-9) -> float:
    """
    区間 [ta, tb] で κ_s(t) ≈ 0 となる t を探す（符号変化点）。
    brentq を使用する。見つからなければ中点を返す。
    """
    fa = curve.signed_curvature(ta)
    fb = curve.signed_curvature(tb)

    # tbが範囲の下限と一致する場合、上限に置き換える
    # 例えば[0, 2π)の曲線で、ta=2π-0.01, tb=0.0のようなケースを考慮
    t_min, t_max = curve.get_range()
    if (ta < t_min - eps) or (tb > t_max + eps):
        raise ValueError(f"ta={ta} or tb={tb} is out of range [{t_min}, {t_max}]")
    if ta > tb:
        tb = t_max + (tb - t_min)

    # ±inf を有限値に置換して符号のみ保存
    def _finite_sign(v):
        if v == float('inf'):
            return 1.0
        elif v == float('-inf'):
            return -1.0
        return np.sign(v)

    # 符号が同じなら細かく探す
    if _finite_sign(fa) == _finite_sign(fb):
        n_sample = 64
        ts = np.linspace(ta, tb, n_sample + 1)
        ks = []
        for t in ts:
            t_eval = t
            if t_eval > t_max:
                t_eval -= (t_max - t_min)
            ks.append(curve.signed_curvature(t_eval))
        ks = np.array(ks)
        for i in range(len(ks) - 1):
            if _finite_sign(ks[i]) != _finite_sign(ks[i + 1]):
                ta, tb = ts[i], ts[i + 1]
                fa, fb = ks[i], ks[i + 1]
                break
        else:
            # 符号変化なし → 中点を返す
            return (ta + tb) / 2.0

    # ±inf の端点がある場合、少しずらして有限値にする
    h = 1e-8 * (tb - ta)
    if not np.isfinite(fa):
        ta = ta + h
    if not np.isfinite(fb):
        tb = tb - h

    try:
        def signed_curvature_wrapper(t, t_max_in=t_max, t_min_in=t_min):
            if t > t_max_in:  # tbのみ上限を超える場合がある
                t -= (t_max_in - t_min_in)
            return curve.signed_curvature(t)
        result = _scipy_optimize.brentq(
            signed_curvature_wrapper,
            ta, tb, xtol=eps * abs(tb - ta), maxiter=200
        )
        return result
    except ValueError:
        return (ta + tb) / 2.0

def _is_convex_outward(curve: CurveBase2D, t: float, eps: float = 1e-9) -> bool:
    """2次元の曲線C上の点C(t)が外側に凸か判定 (通常の意味での凸)"""
    kappa = curve.signed_curvature(t)
    cw = curve.is_clockwise()
    if kappa > eps and not cw:
        return True
    if kappa < -eps and cw:
        return True
    return False


def _is_convex_inward(curve: CurveBase2D, t: float, eps: float = 1e-9) -> bool:
    """2次元の曲線C上の点C(t)が内側に凸か判定 (通常の意味での凹)"""
    kappa = curve.signed_curvature(t)
    cw = curve.is_clockwise()
    if kappa < -eps and not cw:
        return True
    if kappa > eps and cw:
        return True
    return False

def _validate_curve(curve: CurveBase2D) -> None:
    if not curve.is_closed():
        raise ValueError("曲線が閉じていません。閉曲線にのみ対応しています。")
    if curve.has_self_intersection():
        raise ValueError("曲線に自己交差があります。自己交差のない曲線にのみ対応しています。")



#---------------------------------------------------------------------------
# 極値探索
#---------------------------------------------------------------------------

def _dedup(curve: CurveBase2D, pts: list[float],
           period: float, merge_tol: float) -> list[float]:
    """近すぎる点を重複とみなして除去する。

    Parameters
    ----------
    curve : CurveBase2D
        対象の曲線（閉曲線であることが前提）。
    pts : list[float]
        重複除去前の点のリスト（パラメータ値）。
    period : float
        曲線の周期（get_range() の幅）。
    merge_tol : float
        近すぎる点を重複とみなす閾値。例えば 1e-4 を指定すると、周期の 0.01% より近い点は重複とみなされる。
    """
    if not pts:
        return pts
    pts = sorted(pts)
    merged: list[float] = [pts[0]]
    for t in pts[1:]:
        # 通常差分
        gap = t - merged[-1]
        # 閉曲線の折り返し考慮
        if curve.is_closed():
            gap = min(gap, period - gap)
        if gap > merge_tol * period:
            merged.append(t)
    # 閉曲線: 先頭と末尾が周期的に近い場合を除去
    if curve.is_closed() and len(merged) >= 2:
        circ_gap = (merged[0] + period) - merged[-1]
        if circ_gap < merge_tol * period:
            merged = merged[:-1]
    return merged

def _find_curvature_extrema(
    curve: CurveBase2D,
    n_init: int = 500,
    tol: float = 1e-9,
    merge_tol: float = 1e-4) -> tuple[list[float], list[float]]:
    """
    符号付曲率 κ_s(t) の極大・極小点を scipy.optimize.minimize_scalar で探索する。

    アルゴリズム:
      1. t を n_init 点等間隔サンプリングして κ を計算
      2. 隣接3点で局所極大・極小の「ブラケット」を検出
      3. 各ブラケット内で minimize_scalar (bounded) により精密化
      4. 近すぎる解を重複除去

    Returns
    -------
    maxima : 極大点の t リスト
    minima : 極小点の t リスト
    """
    t0, t1 = curve.get_range()
    ts_all = np.linspace(t0, t1, n_init, endpoint=False)

    # 直線部に含まれる点を除外する
    linear_segments = curve.get_linear_segments()
    def _in_linear(t: float) -> bool:
        for us, ue in linear_segments:
            if us <= t <= ue:
                return True
        return False

    ts = np.array([t for t in ts_all if not _in_linear(t)])
    if len(ts) == 0:
        # 全点が直線部に含まれる（完全な多角形など）→ 極値なし
        return [], []

    kappa = np.array([curve.signed_curvature(t) for t in ts])
    n = len(ts)
    period = t1 - t0
    dt = period / n

    maxima: list[float] = []
    minima: list[float] = []

    def kappa_fn(t: float) -> float:
        return curve.signed_curvature(t)

    for i in range(n):
        prev = (i - 1) % n
        nxt  = (i + 1) % n
        k_p, k_i, k_n = kappa[prev], kappa[i], kappa[nxt]

        # 局所極大
        if k_i > k_p and k_i > k_n:
            result = _scipy_optimize.minimize_scalar(
                lambda t: -kappa_fn(t),
                bounds=(ts[i] - dt, ts[i] + dt),
                method="bounded",
                options={"xatol": tol},
            )
            if not isinstance(result, _scipy_optimize.OptimizeResult):
                raise RuntimeError(f"Optimization failed: {result.message}")
            maxima.append(float(result.x))

        # 局所極小
        if k_i < k_p and k_i < k_n:
            result = _scipy_optimize.minimize_scalar(
                kappa_fn,
                bounds=(ts[i] - dt, ts[i] + dt),
                method="bounded",
                options={"xatol": tol},
            )
            if not isinstance(result, _scipy_optimize.OptimizeResult):
                raise RuntimeError(f"Optimization failed: {result.message}")
            minima.append(float(result.x))

    # t を [t0, t1) に正規化（閉曲線）
    def normalize(t: float) -> float:
        while t < t0:
            t += period
        while t >= t1:
            t -= period
        return t

    if curve.is_closed():
        maxima = [normalize(t) for t in maxima]
        minima = [normalize(t) for t in minima]

    # 重複除去
    return (_dedup(curve, maxima, period, merge_tol),
            _dedup(curve, minima, period, merge_tol))



#---------------------------------------------------------------------------
# サンプリング
#---------------------------------------------------------------------------

def _interpolate_closed_points(ts: list[float], t0: float, t1: float,
                               n_vert: int,
                               skip_interval_fn=None) -> list[float]:
    """
    点数が少ない場合、n_vert付近になるよう各間隔を等間隔に補間する。

    skip_interval_fn : callable(ta, tb) -> bool, optional
        True を返す区間は補間をスキップする（直線部の除外に使用）。
    """
    num_pts = len(ts)
    if num_pts == 0:
        return []

    # n_vert を現在の点数で割り、倍率 n を決定する
    n = n_vert / num_pts
    if n <= 1.0:
        return ts

    n_int = int(np.floor(n))
    interpolated_ts = []

    for i in range(num_pts):
        start_t = ts[i]

        # 次の参照点（閉曲線のため末尾の場合は一周させる）
        if i < num_pts - 1:
            end_t = ts[i+1]
            total_gap = end_t - start_t
        else:
            # 閉曲線対応: ts[-1] -> t1 -> t0 -> ts[0]
            first_t = ts[0]
            total_gap = (t1 - start_t) + (first_t - t0)

        # 直線区間はスキップ（始端のみ追加）
        if skip_interval_fn is not None:
            end_t_for_check = ts[i+1] if i < num_pts - 1 else ts[0]
            if skip_interval_fn(start_t, end_t_for_check):
                interpolated_ts.append(start_t)
                continue

        # 各区間を n_int 個に分割して追加
        for j in range(n_int):
            delta = total_gap * j / n_int
            t_val = start_t + delta

            # パラメータ範囲 [t0, t1] を超えた場合の正規化
            if t_val > t1:
                t_val = t0 + (t_val - t1)
            elif t_val < t0: # 念のため
                t_val = t1 - (t0 - t_val)

            interpolated_ts.append(t_val)

    return interpolated_ts

def _insert_corners(curve: CurveBase2D, ts: list[float]) -> list[float]:
    """角点のパラメータ値をサンプル点列に追加して昇順に返す。"""
    corner_params = curve.get_corner_params()
    t0, t1 = curve.get_range()
    if not corner_params:
        return ts
    ts_set = set(ts)
    for tc in corner_params:
        if t0 <= tc < t1 and tc not in ts_set:
            ts.append(tc)
    ts.sort()
    return ts

def _insert_linear_endpoints(linear_segments: list[tuple[float, float]],
                             ts: list[float]) -> list[float]:
    """直線部の始端・終端をサンプル点列に追加して昇順に返す。"""
    if not linear_segments:
        return ts
    ts_set = set(ts)
    for us, ue in linear_segments:
        if us not in ts_set:
            ts.append(us)
        if ue not in ts_set:
            ts.append(ue)
    ts.sort()
    return ts

def _in_linear(linear_segments: list[tuple[float, float]],
               t: float, eps: float = 1e-9) -> bool:
    """パラメータ t が直線部の内部（端点を除く）に含まれるか判定する。"""
    for us, ue in linear_segments:
        if us + eps < t < ue - eps:
            return True
    return False

def _is_linear_interval(linear_segments: list[tuple[float, float]],
                        ta: float, tb: float, eps: float = 1e-9) -> bool:
    """区間 [ta, tb] が直線部の始端・終端と一致するか判定する。"""
    for us, ue in linear_segments:
        if abs(ta - us) < eps and abs(tb - ue) < eps:
            return True
    return False

def _sample_points(curve: CurveBase2D, n_vert: int) -> tuple[list[float], list[np.ndarray]]:
    """パラメータ範囲を分割するサンプル点を返す。

    角点がある場合、角点のパラメータ値をサンプル点列に追加し、パラメータ値で
    昇順にソートする。これにより、角点は必ずサンプル点として含まれ、凸性の判定や
    接線処理が正しく行われることが保証される。

    直線部がある場合、直線部の始端・終端を強制挿入したうえで、直線部内部への
    サンプル点生成を行わない。これにより、不要な分割が抑制される。
    """
    t0, t1 = curve.get_range()
    linear_segments = curve.get_linear_segments()

    # 符号付き曲率の極大/極小点を取得し、存在する場合はそれをサンプル点とする
    maxima, minima = _find_curvature_extrema(curve)
    if maxima or minima:
        print(f"  Found curvature extrema: {len(maxima)} maxima, {len(minima)} minima")
        ts = sorted(set(maxima + minima))

        # 直線部を考慮した補間：直線区間には点を追加しない
        ts = _interpolate_closed_points(
            ts, t0, t1, n_vert,
            skip_interval_fn=lambda ta, tb: _is_linear_interval(linear_segments, ta, tb)
        )
        ts = _insert_corners(curve, ts)
        ts = _insert_linear_endpoints(linear_segments, ts)
        print(f"  Sampled t values: {[f'{t:.3f}' for t in ts]}")
        pts = [curve.evaluate(t) for t in ts]
        return ts, pts

    # 極値が見つからない場合は等間隔サンプリング（直線部内部は除外）
    ts_all = [t0 + (t1 - t0) * i / n_vert for i in range(n_vert)]
    ts = [t for t in ts_all if not _in_linear(linear_segments, t)]
    ts = _insert_corners(curve, ts)
    ts = _insert_linear_endpoints(linear_segments, ts)
    print(f"  Sampled t values: {[f'{t:.3f}' for t in ts]}")
    pts = [curve.evaluate(t) for t in ts]
    return ts, pts


def _classify_points(curve: CurveBase2D, ts: list[float],
                     circumscribed: bool, eps: float = 1e-9) -> list[bool]:
    """
    各点が頂点(False)か接点(True)かを分類する。
    circumscribed=True: 外包 → 内側に凸なら頂点(False)、外側に凸なら接点(True)
    circumscribed=False: 内包 → 外側に凸なら頂点(False)、内側に凸なら接点(True)
    """
    is_contact = []
    for t in ts:
        if circumscribed:
            # 外包: 内側に凸(inn) → 頂点(False), 外側に凸(out) → 接点(True)
            inn = _is_convex_inward(curve, t, eps)
            if inn:
                is_contact.append(False)
            else:
                is_contact.append(True)
        else:
            # 内包: 外側に凸(out) → 頂点(False), 内側に凸(inn) → 接点(True)
            out = _is_convex_outward(curve, t, eps)
            if out:
                is_contact.append(False)
            else:
                is_contact.append(True)
    return is_contact


def _build_polygon_vertices(curve: CurveBase2D,
                             ts: list[float],
                             pts: list[np.ndarray],
                             is_contact: list[bool]) -> PolygonData:
    """
    隣接ペアのパターンに従って多角形の頂点列を構築する。
    """
    n = len(ts)
    polygon: list[np.ndarray] = []
    on_curve_lst: list[bool] = []
    curve_params_lst: list[float] = []
    linear_segments = curve.get_linear_segments()

    def add(pt: np.ndarray, on_curve: bool, param: float = 0.0) -> None:
        polygon.append(pt)
        on_curve_lst.append(on_curve)
        curve_params_lst.append(param if on_curve else 0.0)

    def _is_linear_pair(ta: float, tb: float, eps: float = 1e-9) -> bool:
        """ta と tb が同一直線部の始端・終端であるかを判定する。"""
        for us, ue in linear_segments:
            if abs(ta - us) < eps and abs(tb - ue) < eps:
                return True
        return False

    for i in range(n):
        j = (i + 1) % n
        ta, tb = ts[i], ts[j]
        Pa = pts[i]
        Pb = pts[j]
        ca = is_contact[i]
        cb = is_contact[j]

        # 直線区間ペアの場合：Pa と Pb を直線で結ぶだけ（曲率零点探索は行わない）
        if _is_linear_pair(ta, tb):
            add(Pa, on_curve=True, param=ta)
            continue

        # 接線方向（角点では二等分接線が自動的に使われる）
        da = curve.tangent(ta)   # Pa → Pb 方向の接線
        db = curve.tangent(tb)   # Pb → Pa 方向の接線（反転）

        if not ca and not cb:
            # 両方頂点 → 線分（Pa を追加、次のループで Pb が追加される）
            add(Pa, on_curve=True, param=ta)

        elif ca and cb:
            # 両方接点 → 交点を頂点とする
            # Pa からは Pb に向かう接線、Pb からは Pa に向かう接線
            vertex = _line_intersect(Pa, da, Pb, -db)
            add(Pa,     on_curve=True,  param=ta)  # 接点 Pa（多角形の辺の途中点として記録）
            add(vertex, on_curve=False)             # 交点が多角形の頂点

        elif not ca and cb:
            # Pa が頂点、Pb が接点
            t_mid = _find_zero_curvature(curve, ta, tb)
            Pm = curve.evaluate(t_mid)
            dm = curve.tangent(t_mid)
            # Pa → Pm の線分
            vertex = _line_intersect(Pm, dm, Pb, -db)
            add(Pa,     on_curve=True,  param=ta)
            add(Pm,     on_curve=True,  param=t_mid)
            # Pm から Pb 方向の接線と、Pb から Pa 方向の接線の交点
            add(vertex, on_curve=False)

        else:
            # Pa が接点、Pb が頂点
            t_mid = _find_zero_curvature(curve, ta, tb)
            Pm = curve.evaluate(t_mid)
            dm = curve.tangent(t_mid)
            # Pb → Pm の線分
            vertex = _line_intersect(Pm, -dm, Pa, da)
            add(Pa,     on_curve=True,  param=ta)
            # Pm から Pa 方向の接線と、Pa から延びる接線の交点
            add(vertex, on_curve=False)
            add(Pm,     on_curve=True,  param=t_mid)

    return PolygonData(
        vertices=np.array(polygon),
        on_curve=on_curve_lst,
        curve_params=curve_params_lst,
    )

def _compute_extremal_polygon(curve: CurveBase2D, n_vert: int, circumscribed: bool,
                              eps: float = 1e-9) -> PolygonData:
    """
    外包/内包多角形の頂点列を計算する（circumscribed=True で外包、False で内包）。

    Parameters
    ----------
    curve : CurveBase2D
        閉曲線（自己交差なし）。角点を含んでもよい。
    n_vert : int
        初期分割数（実際の頂点数はこれより多くなる場合あり; 最大で2倍強）
    circumscribed : bool
        True なら外包多角形、False なら内包多角形を構築する。
    eps : float
        曲率判定の閾値

    Returns
    -------
    PolygonData
        多角形の頂点列・曲線上フラグ・曲線パラメータ
    """
    st_1 = _time.perf_counter()
    _validate_curve(curve)
    st_2 = _time.perf_counter()
    ts, pts = _sample_points(curve, n_vert)
    st_3 = _time.perf_counter()
    is_contact = _classify_points(curve, ts, circumscribed=circumscribed, eps=eps)
    st_4 = _time.perf_counter()
    result = _build_polygon_vertices(curve, ts, pts, is_contact)
    et_4 = _time.perf_counter()

    print(f"  COMPUTATION TIME ({'circumscribed' if circumscribed else 'inscribed'}):")
    print(f"    validation    : {st_1-st_2:.4f} sec")
    print(f"    sampling      : {st_2-st_3:.4f} sec")
    print(f"    classification: {st_3-st_4:.4f} sec")
    print(f"    polygonize    : {st_4-et_4:.4f} sec")
    return result

def compute_circumscribed_polygon(curve: CurveBase2D,
                                  n_vert: int,
                                  eps: float = 1e-9) -> PolygonData:
    """
    外包多角形（曲線を完全に囲む多角形）の頂点列を計算する。

    Parameters
    ----------
    curve : CurveBase2D
        閉曲線（自己交差なし）。角点を含んでもよい。
    n_vert : int
        初期分割数（実際の頂点数はこれより多くなる場合あり; 最大で2倍強）
    eps : float
        曲率判定の閾値

    Returns
    -------
    PolygonData
        多角形の頂点列・曲線上フラグ・曲線パラメータ
    """
    return _compute_extremal_polygon(curve, n_vert, circumscribed=True, eps=eps)


def compute_inscribed_polygon(curve: CurveBase2D,
                              n_vert: int,
                              eps: float = 1e-9) -> PolygonData:
    """
    内包多角形（曲線に内包される多角形）の頂点列を計算する。

    Parameters
    ----------
    curve : CurveBase2D
        閉曲線（自己交差なし）。角点を含んでもよい。
    n_vert : int
        初期分割数（実際の頂点数はこれより多くなる場合あり; 最大で2倍強）
    eps : float
        曲率判定の閾値

    Returns
    -------
    PolygonData
        多角形の頂点列・曲線上フラグ・曲線パラメータ
    """
    return _compute_extremal_polygon(curve, n_vert, circumscribed=False, eps=eps)
