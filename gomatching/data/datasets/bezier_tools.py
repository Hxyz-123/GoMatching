import numpy as np
import cv2
from scipy.special import comb as n_over_k

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def bezier_points(p1, p2, num_points):
    pts = []
    pts.append(p1)
    for i in range(1, num_points+1):
        t = i / (num_points+1)
        x = int((1-t)*p1[0] + t*p2[0])
        y = int((1-t)*p1[1] + t*p2[1])
        pts.append([x, y])
    pts.append(p2)
    return pts

def longest_edges(pts_arr):
    poly = pts_arr
    edges = [(poly[i], poly[(i+1) % 4]) for i in range(4)]
    edges_sorted = sorted(edges, key=lambda e: -distance(*e))
    return edges_sorted[:2]

def cpt_bezier_pts(rect):
    rect = np.array(rect)
    edges = longest_edges(rect)
    bezier_pts = []
    for edge in edges:
        bezier_pts.extend(bezier_points(*edge, 2))
    bzr_arr = np.array(bezier_pts)
    bzr_clk = bzr_arr
    return bzr_clk

def polygon2rbox(poly, image_height, image_width):
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1,2)
    pts = pts.tolist()
    return pts

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]


class Bezier():
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Bezier, self).__init__()
        self.x1 = ctps[0]
        self.x2 = ctps[2]
        self.y1 = ctps[1]
        self.y2 = ctps[3]
        self.x0 = ps[0, 0]
        self.x3 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y3 = ps[-1, 1]
        self.inner_ps = np.array(ps[1:-1, :], dtype=float)
        self.t = np.linspace(0, 1, 81)

    def __call__(self):
        x0, x1, x2, x3, y0, y1, y2, y3 = self.control_points()
        t = self.t
        bezier_x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
                (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
        bezier_y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * (
                (1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
        bezier = np.stack((bezier_x, bezier_y), axis=1)
        diffs = np.expand_dims(bezier, axis=0) - np.expand_dims(self.inner_ps, axis=1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.y0, self.y1, self.y2, self.y3


def train(x, y, ctps):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    bezier = Bezier(ps, ctps)

    return bezier.control_points()


def bezier_fit(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2) ** 0.5
    t = dt / dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :].flatten().tolist()
    return medi_ctp


def polygon_to_bezier_pts(polygons):
    assert len(polygons) % 2 == 0
    mid_idx = len(polygons) // 2

    curve_data_top = polygons[:mid_idx]
    curve_data_bottom = polygons[mid_idx:]

    x_data = curve_data_top[:, 0]
    y_data = curve_data_top[:, 1]

    init_control_points = bezier_fit(x_data, y_data)

    x0, x1, x2, x3, y0, y1, y2, y3 = train(x_data, y_data, init_control_points)

    x_data_b = curve_data_bottom[:, 0]
    y_data_b = curve_data_bottom[:, 1]

    init_control_points_b = bezier_fit(x_data_b, y_data_b)


    x0_b, x1_b, x2_b, x3_b, y0_b, y1_b, y2_b, y3_b = train(x_data_b, y_data_b, init_control_points_b)
    control_points = np.array([
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x0_b, y0_b],
        [x1_b, y1_b],
        [x2_b, y2_b],
        [x3_b, y3_b]
    ])
    return control_points