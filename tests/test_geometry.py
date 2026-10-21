# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import pytest

from pinnx import geometry as g

GEOMETRIES = {
    "interval": lambda: g.Interval(0.0, 1.0),
    "rectangle": lambda: g.Rectangle([0.0, 0.0], [1.0, 2.0]),
    "disk": lambda: g.Disk([0.0, 0.0], 1.0),
    "ellipse": lambda: g.Ellipse([0.0, 0.0], 2.0, 1.0),
    "star": lambda: g.StarShaped([0.0, 0.0], 1.0, [0.2], [-0.1]),
    "triangle": lambda: g.Triangle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
    "polygon": lambda: g.Polygon([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]),
    "cuboid": lambda: g.Cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    "sphere": lambda: g.Sphere([0.0, 0.0, 0.0], 1.0),
    "hypercube": lambda: g.Hypercube([0.0] * 4, [1.0] * 4),
    "hypersphere": lambda: g.Hypersphere([0.0] * 4, 1.0),
    "union": lambda: g.Rectangle([0.0, 0.0], [1.0, 1.0]) | g.Disk([1.0, 0.5], 0.5),
    "difference": lambda: g.Rectangle([0.0, 0.0], [1.0, 1.0]) - g.Disk([0.5, 0.5], 0.25),
    "intersection": lambda: g.Rectangle([0.0, 0.0], [1.0, 1.0]) & g.Disk([0.0, 0.0], 1.0),
    "interval_difference": lambda: g.Interval(0.0, 1.0) - g.Interval(0.2, 0.4),
}


@pytest.fixture(params=sorted(GEOMETRIES))
def geom(request):
    return GEOMETRIES[request.param]()


@pytest.mark.parametrize("random", ["pseudo", "Hammersley"])
def test_random_points_are_inside(geom, random):
    x = geom.random_points(64, random=random)
    assert np.shape(x) == (64, geom.dim)
    assert np.all(np.asarray(geom.inside(x)))


def test_random_boundary_points(geom):
    x = geom.random_boundary_points(32)
    assert np.shape(x) == (32, geom.dim)
    assert np.all(np.asarray(geom.on_boundary(x)))
    try:
        n = np.asarray(geom.boundary_normal(x))
    except NotImplementedError:
        return
    assert n.shape == (32, geom.dim)
    assert np.issubdtype(n.dtype, np.floating)


def test_uniform_points(geom):
    try:
        x = geom.uniform_points(64)
    except NotImplementedError:
        pytest.skip("uniform_points not implemented")
    assert np.shape(x)[1] == geom.dim
    assert np.all(np.asarray(geom.inside(x)))


def test_uniform_boundary_points(geom):
    try:
        x = geom.uniform_boundary_points(64)
    except NotImplementedError:
        pytest.skip("uniform_boundary_points not implemented")
    assert np.shape(x)[1] == geom.dim
    assert np.all(np.asarray(geom.on_boundary(x)))


def test_on_boundary_is_elementwise(geom):
    x = geom.random_points(8)
    assert np.shape(geom.on_boundary(x)) == (8,)


def test_bbox_contains_samples(geom):
    lo, hi = (np.asarray(b) for b in geom.bbox)
    x = np.asarray(geom.random_points(256))
    assert np.all(x >= lo - 1e-6) and np.all(x <= hi + 1e-6)


def test_star_bbox_with_negative_coefficients():
    lo, hi = g.StarShaped([0.0, 0.0], 1.0, [-0.3], [0.0]).bbox
    assert np.all(np.asarray(hi) >= 1.3 - 1e-6)
    assert np.all(np.asarray(lo) <= -1.3 + 1e-6)


def test_ellipse_bbox_and_diameter():
    e = g.Ellipse([0.0, 0.0], 2.0, 1.0)
    np.testing.assert_allclose(np.asarray(e.bbox[1]), [2.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(e.bbox[0]), [-2.0, -1.0], atol=1e-5)
    assert float(e.diam) == pytest.approx(4.0)


def test_rotated_ellipse_bbox():
    e = g.Ellipse([0.0, 0.0], 2.0, 1.0, angle=np.pi / 2)
    np.testing.assert_allclose(np.asarray(e.bbox[1]), [1.0, 2.0], atol=1e-5)


@pytest.mark.parametrize(
    "geom_fn, x, component, expected",
    [
        (lambda: g.Interval(0.0, 1.0), [[0.0], [0.5]], 0, [[1.0], [0.5]]),
        (lambda: g.Rectangle([0.0, 0.0], [1.0, 2.0]), [[0.0, 0.5], [1.0, 2.0]], 0, [[1.0, 0.5], [0.0, 2.0]]),
        (lambda: g.Rectangle([0.0, 0.0], [1.0, 2.0]), [[0.5, 0.0]], 1, [[0.5, 2.0]]),
    ],
)
def test_periodic_point(geom_fn, x, component, expected):
    y = geom_fn().periodic_point(np.asarray(x), component)
    np.testing.assert_allclose(np.asarray(y), expected)


def test_hypercube_vertex_normal_is_normalized():
    n = np.asarray(g.Rectangle([0.0, 0.0], [1.0, 1.0]).boundary_normal(np.array([[0.0, 0.0], [0.5, 1.0]])))
    np.testing.assert_allclose(n, [[-(0.5 ** 0.5), -(0.5 ** 0.5)], [0.0, 1.0]], rtol=1e-6)


def test_dict_geometry_normals_are_dimensionless():
    import saiunit as u
    geom = g.Rectangle([0.0, 0.0], [1.0, 1.0]).to_dict_point(x=u.meter, y=u.meter)
    pts = {"x": np.array([0.0, 0.5]) * u.meter, "y": np.array([0.5, 1.0]) * u.meter}
    n = geom.boundary_normal(pts)
    for v in n.values():
        assert not isinstance(v, u.Quantity) or v.dim.is_dimensionless
    np.testing.assert_allclose(np.asarray(n["x"]), [-1.0, 0.0])
    np.testing.assert_allclose(np.asarray(n["y"]), [0.0, 1.0])


def test_csg_periodic_point():
    geom = g.Rectangle([0.0, 0.0], [1.0, 1.0]) | g.Rectangle([1.0, 0.0], [2.0, 1.0])
    y = geom.periodic_point(np.array([[0.0, 0.5]]), 0)
    assert np.shape(y) == (1, 2)


@pytest.mark.parametrize("where", [None, "left", "right"])
@pytest.mark.parametrize("smoothness", ["C0", "C0+", "Cinf"])
def test_interval_boundary_constraint_factor(where, smoothness):
    f = g.Interval(0.0, 1.0).boundary_constraint_factor(np.array([[0.0], [0.5], [1.0]]), smoothness, where)
    assert np.all(np.isfinite(np.asarray(f)))


@pytest.mark.parametrize("smoothness", ["C0", "C0+", "Cinf"])
def test_hypercube_boundary_constraint_factor(smoothness):
    geom = g.Hypercube([0.0] * 3, [1.0] * 3)
    x = np.array([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]])
    f = np.asarray(geom.boundary_constraint_factor(x, smoothness))
    assert f.shape == (2, 1)
    assert f[0, 0] > 0 and f[1, 0] == pytest.approx(0)


def test_polygon_random_boundary_points_on_edges():
    poly = GEOMETRIES["polygon"]()
    x = poly.random_boundary_points(200)
    assert np.all(np.asarray(poly.on_boundary(x)))


def test_polygon_boundary_normal_float_when_first_point_interior():
    poly = GEOMETRIES["polygon"]()
    x = np.array([[0.5, 0.5], [0.5, 0.0]])
    np.testing.assert_allclose(np.asarray(poly.boundary_normal(x)), [[0.0, 0.0], [0.0, -1.0]])


def test_pointcloud_boundary_normal_order():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    bpts = np.array([[0.0, 0.0], [1.0, 0.0]])
    normals = np.array([[-1.0, 0.0], [1.0, 0.0]])
    pc = g.PointCloud(pts, boundary_points=bpts, boundary_normals=normals)
    x = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(pc.boundary_normal(x)), [[1.0, 0.0], [-1.0, 0.0]])


def test_time_domain_and_geometry_x_time():
    geomtime = g.GeometryXTime(g.Interval(0.0, 1.0), g.TimeDomain(0.0, 1.0))
    x = geomtime.random_points(16)
    assert np.all(np.asarray(geomtime.inside(x)))
    x = geomtime.random_initial_points(8)
    assert np.all(np.asarray(geomtime.on_initial(x)))
    x = geomtime.random_boundary_points(8)
    assert np.all(np.asarray(geomtime.on_boundary(x)))


@pytest.mark.parametrize("name", ["interval", "rectangle", "disk", "triangle", "polygon", "cuboid", "sphere", "union"])
def test_boundary_normal_is_jittable(name):
    import jax

    geom = GEOMETRIES[name]()
    x = np.asarray(geom.random_boundary_points(8))
    np.testing.assert_allclose(
        np.asarray(jax.jit(geom.boundary_normal)(x)), np.asarray(geom.boundary_normal(x)), atol=1e-5
    )
