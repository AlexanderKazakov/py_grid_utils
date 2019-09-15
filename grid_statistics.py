from utils import *


fname = 'grids/results/9_percent/skull.vtk'
vtk_grid = readVtkGrid(fname)
points, cells, materials = convertVtkGridToNumpy(vtk_grid, 4)
materials = materials[0]
min_height, asp_ratio = grid_quality(points, cells)

writeVtkGrid(constructVtkGrid(points, cells, {
    'material': ('int', materials),
    'log10_min_h': ('float', np.log10(min_height)),
    'asp_ratio': ('float', asp_ratio),
}), 'grids/res.vtk')

bound_facets = findBoundFacets(np.c_[cells, materials])
different_material = np.where(np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1))[0]
bound_facets = np.delete(bound_facets, different_material, axis=0)
bound_facets = bound_facets[:, :-1]
new_points, new_facets = remove_unused_points(points, bound_facets)
surface_grid = constructVtkGrid(points, bound_facets)
writeVtkGrid(surface_grid, "grids/surfaces.vtk")
to_stl('grids/surfaces.vtk')



