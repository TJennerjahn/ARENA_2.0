#%%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


#%%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    result = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.ones(num_pixels, out=result[:, 1, 0])
    t.linspace(-y_limit, y_limit, num_pixels, out=result[:, 1, 1])
    return result


rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)

#%%
if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})


#%%
# Exercise: Check which of these line segments intersect with the rays from earlier.
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

render_lines_with_plotly([*segments, *rays1d])

#%%
@jaxtyped
def intersect_ray_1d(ray: Float[Tensor, "points=2 dim=3"], segment: Float[Tensor, "points=2 dim=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    ray = ray[..., :2]
    segment = segment[..., :2]

    O, D = ray
    L_1, L_2 = segment

    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False
    u, v = sol
    return u >= 0 and 0 <= v <= 1
    


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

#%%
x = t.randn(2, 3)
x_repeated = einops.repeat(x, 'a b -> a b c', c=4)
print("x:", x)
print("x_repeated:", x_repeated)

assert x_repeated.shape == (2, 3, 4)
for c in range(4):
    t.testing.assert_close(x, x_repeated[:, :, c])

#%%
A = t.randint(10, (2, 3))  # Random integers from 0 to 2, shape (2, 3)
B = t.randint(3, (3,))  # Random integers from 0 to 3, shape (3,)
print("A:", A)
print("B:", B)
AplusB = A + B
print("AplusB:", AplusB)

assert AplusB.shape == (2, 3)
for i in range(2):
    t.testing.assert_close(AplusB[i], A[i] + B)

#%%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    rays = rays[..., :2]
    segments = segments[..., :2]

    rays = einops.repeat(rays, 'rays points coordinates -> rays segments points coordinates', segments=segments.shape[0])
    segments = einops.repeat(segments, 'segments points coordinates -> rays segments points coordinates', rays=rays.shape[0])
    # print("rays.shape:", rays.shape)
    # print("segments.shape:", segments.shape)

    O, D = rays[:,:,0], rays[:,:,1]
    L_1, L_2 = segments[:,:,0], segments[:,:,1]

    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O
    mask = t.linalg.det(mat).abs() < 1e-6
    mat[mask] = t.eye(2)

    # print("origins.shape:", O.shape)
    # print("directions.shape:", D.shape)
    # print("mat.shape:", mat.shape)
    # print("vec.shape:", vec.shape)
    # print("mask.shape", mask.shape)

    try:
        sol = t.linalg.solve(mat, vec)

    except t.linalg.LinAlgError as er:
        print("er:", er)
        return False
    print("sol.shape:", sol.shape)

    u, v = sol[..., 0], sol[..., 1]
    # print("u.shape:", u.shape)
    # print("v.shape:", v.shape)
    result = (u >= 0) & (0 <= v) & (v <= 1)
    result[mask] = False 
    result = result.any(dim=-1)
    # print("result.shape:", result.shape)

    return result


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

#%%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''

    result = t.zeros((num_pixels_y * num_pixels_z, 2, 3), dtype=t.float32)
    t.ones(num_pixels_y * num_pixels_z, out=result[:, 1, 0])
    result[:, 1, 1] = einops.repeat(t.linspace(-y_limit, y_limit, num_pixels_y), 'y -> (y z)', z=num_pixels_z)
    result[:, 1, 2] = einops.repeat(t.linspace(-z_limit, z_limit, num_pixels_z), 'z -> (y z)', y=num_pixels_y)
    return result


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)

#%%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)

#%%
Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    mat = t.stack([-D, B - A, C - A], dim=-1)
    vec = O - A
    try:
        sol = t.linalg.solve(mat, vec)
    except t.linalg.LinAlgError as er:
        print("er:", er)
        return False
    
    s, u, v = sol
    return (s >= 0) & (0 <= u) & (0 <= v) & (u + v <= 1)


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)


#%%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    A,B,C = einops.repeat(triangle, 'trianglePoints dims -> trianglePoints nrays dims', nrays=rays.shape[0])
    O, D = rays.unbind(dim=1)
    mat = t.stack([-D, B - A, C - A], dim=-1)
    det = t.linalg.det(mat)
    is_singular = det.abs() < 1e-6
    mat[is_singular] = t.eye(3)
    vec = O - A
    sol = t.linalg.solve(mat, vec)
    
    s, u, v = sol.unbind(dim=-1)
    result = (s >= 0) & (0 <= u) & (0 <= v) & (u + v <= 1) & (~is_singular)
    return result


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 10
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

#%%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

#%%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    rays = einops.repeat(rays, 'nrays rayPoints dims -> nrays ntriangles rayPoints dims', ntriangles=triangles.shape[0])
    triangles = einops.repeat(triangles, 'ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims', nrays=rays.shape[0])
    print("rays.shape:", rays.shape)
    print("triangles.shape:", triangles.shape)
    A,B,C = triangles.unbind(dim=2)
    O, D = rays.unbind(dim=2)
    mat = t.stack([-D, B - A, C - A], dim=-1)
    det = t.linalg.det(mat)
    is_singular = det.abs() < 1e-6
    mat[is_singular] = t.eye(3)
    vec = O - A
    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)
    intersects = (s >= 0) & (0 <= u) & (0 <= v) & (u + v <= 1) & (~is_singular)
    s[~intersects] = t.inf
    print("sol.shape:", sol.shape)
    return s.min(dim=1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()


#%%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
