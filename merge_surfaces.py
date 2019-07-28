from utils import *


file1 = 'grids/outer_surface.vtk'
file2 = 'grids/corrected_inner_surfaces.vtk'

points1, facets1, _ = convertVtkGridToNumpy(readVtkGrid(file1))
points2, facets2, _ = convertVtkGridToNumpy(readVtkGrid(file2))

points = np.r_[points1, points2]
facets2 += len(points1)
facets = np.r_[facets1, facets2]

surface_id = np.r_[np.ones(len(facets1)), 2 * np.ones(len(facets2))].astype(np.int64)

vtk_grid = constructVtkGrid(
    points, facets,
    cell_info={'surface_id': ('int', surface_id)})
writeVtkGrid(vtk_grid, 'grids/merged.vtk')
surface_quality(points, facets, draw_picture=True)
to_stl('grids/merged.vtk')

