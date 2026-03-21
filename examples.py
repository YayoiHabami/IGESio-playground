import argparse
import os

from py_igesio.entities.interfaces import CurveBase2D
from py_igesio.entities.curves import EllipseCurve, StarCurve, NURBSCurve, PolylineCurve2D
from py_igesio.py_demo.vis_polygons import vis_containment_polygons

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def get_ellipse_curve() -> EllipseCurve:
    """楕円曲線の例を返す"""
    return EllipseCurve(a=3.0, b=1.5)

def get_star_curve() -> StarCurve:
    """星形曲線の例を返す"""
    return StarCurve(r=2.0, a=0.8, n=5)

def get_nurbs_curve() -> NURBSCurve:
    """NURBS曲線の例を返す"""
    control_points: list[tuple[float, float, float]] = [
        (-5, -3, 0),
        (-2, -6, 0),
        ( 6,  1, 0),
        (-3, -1, 0),
        (-1,  3, 0),
        (-6,  3, 0),
        (-5, -3, 0),   # 最初の点と一致 → 閉曲線
    ]
    weights: list[float] = [1.0] * len(control_points)
    knot_vector: list[float] = [0, 0, 0, 0, 0.1, 0.5, 0.8, 1, 1, 1, 1]
    degree: int = 3

    return NURBSCurve(
        control_points=control_points,
        weights=weights,
        knot_vector=knot_vector,
        degree=degree,
    )

def get_polyline_2d() -> PolylineCurve2D:
    """2D折れ線の例を返す"""
    return PolylineCurve2D([
        [-8, -4], [-8, 6], [-2, 5], [4, 4],  # この3点は一直線
        [3, 2], [5, -2], [0, -2], [-1, -6], [-8, -4]
    ])

def get_curves() -> list[CurveBase2D]:
    """テスト用の曲線のリストを返す"""
    return [
        get_ellipse_curve(),
        get_star_curve(),
        get_nurbs_curve(),
        get_polyline_2d(),
    ]

def demo_containment_polygons(args: argparse.Namespace):
    """複数の曲線に対して内包・外包多角形を計算し、描画するデモ関数"""
    output_dir = os.path.join(OUTPUT_DIR, "containment_polygons") if args.save_fig else None
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for curve in get_curves():
        vis_containment_polygons(
            curve=curve,
            n_vert_circ=args.n_circ, n_vert_insc=args.n_insc,
            distance_eps=args.eps_dist, output_dir=output_dir,
        )

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description="Polygon computation and drawing tool")

    parser.add_argument(
        "example",
        type=str,
        choices=["CONTAINMENT_POLY"],
        help="Type of example to run"
    )
    parser.add_argument(
        "--n_circ",
        type=int,
        default=20,
        help="Number of vertices for circumscribed polygon of curves C(t) (default: 20)"
    )
    parser.add_argument(
        "--n_insc",
        type=int,
        default=20,
        help="Number of vertices for inscribed polygon of curves C(t) (default: 20)"
    )
    parser.add_argument(
        "--eps_dist",
        type=float,
        default=1e-3,
        help="Distance threshold for polygon approximation of curves C(t) (default: 1e-3)"
    )
    parser.add_argument(
        "--save_fig",
        type=lambda x: x.lower() != "false",
        default=True,
        help="Whether to save the figure as an image. If false, a window will open (default: true)"
    )

    return parser.parse_args()

def main():
    """コマンドライン引数を解析して、指定されたデモ関数を実行する"""
    args = parse_args()
    if args.example == "CONTAINMENT_POLY":
        demo_containment_polygons(args)
if __name__ == "__main__":
    main()
