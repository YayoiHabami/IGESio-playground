"""
閉曲線C(t)に対して、内包・外包多角形と近似多角形を計算し、描画するデモ用の関数

NOTE: Python用の実装 (C++への移植は行わない)
"""
import time
from os import path as os_path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from py_igesio.entities.interfaces import CurveBase2D
from py_igesio.numerics.polygons import PolygonData, CurveContainmentPolygons
from py_igesio.entities.curves.algorithms import (
    compute_circumscribed_polygon,
    compute_inscribed_polygon,
    compute_approximate_polygon
)

# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

def _plot_polygon(curve: CurveBase2D,
                  poly_containment: Optional[CurveContainmentPolygons],
                  n_curve_pts: int = 500,
                  title: str = "", save_path: Optional[str] = None) -> None:
    """
    曲線と外包/内包多角形を描画する。

    Parameters
    ----------
    curve : CurveBase2D
    poly_containment : Optional[CurveContainmentPolygons]
        内包・外包多角形と近似多角形のデータ。None の場合は曲線のみ描画する。
    n_curve_pts : 曲線の描画点数
    title : グラフのタイトル
    save_path : Optional[str]
        グラフを保存する場合のファイルパス。None の場合は保存せず表示のみ行う。
    """
    t0, t1 = curve.get_range()
    ts = np.linspace(t0, t1, n_curve_pts)
    curve_pts = np.array([curve.evaluate(t) for t in ts])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")

    # 曲線
    ax.plot(curve_pts[:, 0], curve_pts[:, 1],
            color="steelblue", linewidth=2, label="Curve")

    def _plot_polygon(polygon: PolygonData,
                      label: str, color: str, line_style: str, scale: float = 1.0):
        size = scale * 40
        line_width = scale
        closed = np.vstack([polygon.vertices, polygon.vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1],
                color=color, linewidth=line_width, linestyle=line_style, label=label)
        ax.scatter(polygon.vertices[:, 0], polygon.vertices[:, 1],
                   color=color, s=size, zorder=5)

    if poly_containment is not None:
        # 外包多角形 (内包の後ろに描画するため、点と線を大きく描画する)
        _plot_polygon(poly_containment.circumscribed, "Circumscribed", "tomato", "--", 2)

        # 内包多角形
        _plot_polygon(poly_containment.inscribed, "Inscribed", "forestgreen", "-.", 1)

        # 近似多角形
        _plot_polygon(poly_containment.approximate, "Approximation", "#CE2B4988", "-.", 0.5)

    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

# ---------------------------------------------------------------------------
# テスト関数
# ---------------------------------------------------------------------------

# 近似曲線の許容値を大きめに設定しているが、実際の処理ではおそらく1e-6や1e-8くらいを設定する

def vis_containment_polygons(
        curve: CurveBase2D,
        n_vert_circ: int, n_vert_insc: int,
        distance_eps: float = 1e-3,
        output_dir: Optional[str] = None) -> None:
    """任意のCurveBase2Dに対して外包・内包多角形をテストする。"""
    print(f"=== {curve.get_name()} Test ===")
    st_c = time.perf_counter()
    poly_c = compute_circumscribed_polygon(curve, n_vert_circ)

    st_i = time.perf_counter()
    poly_i = compute_inscribed_polygon(curve, n_vert_insc)

    st_a = time.perf_counter()
    poly_a = compute_approximate_polygon(curve, poly_i, poly_c, distance_eps, 100)
    et_a = time.perf_counter()

    poly_containment = CurveContainmentPolygons(
        circumscribed=poly_c,
        inscribed=poly_i,
        approximate=poly_a
    )

    print(f"  外包多角形 頂点数: {poly_c.count()} ({st_c - st_i} sec)")
    print(f"  内包多角形 頂点数: {poly_i.count()} ({st_i - st_a} sec)")
    print(f"  近似多角形 頂点数: {poly_a.count()} ({st_a - et_a} sec)")

    title = f"{curve.get_name()}, n_vert_circ={n_vert_circ}, n_vert_insc={n_vert_insc}"
    save_path = None
    if output_dir is not None:
        filename = f"{curve.get_name()}_circ{n_vert_circ}_insc{n_vert_insc}.png"
        save_path = os_path.join(output_dir, filename)

    _plot_polygon(curve, poly_containment,
                  title=title, save_path=save_path)
