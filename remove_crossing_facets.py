from utils import *


# the surface to exclude facets outer or crossing the non-convex hull
points, facets, _ = convertVtkGridToNumpy(readVtkGrid("grids/inner_surfaces.vtk"))
facets = np.sort(facets, axis=1)
print(points.shape, facets.shape)

# tetra grid which represents the non-convex hull
grid = readVtkGrid("grids/results/grid_from_outer_surface_015.vtk")
locator = vtk.vtkCellLocator()
locator.SetDataSet(grid)
locator.BuildLocator()

grid_points, grid_cells, _ = convertVtkGridToNumpy(grid)
cells_neighbors = find_cells_neighbors(grid_cells)
border_cells_ids = np.where(np.any(cells_neighbors == -1, axis=1))[0]
for i in range(5):
    border_cells_neighbors = np.unique(cells_neighbors[border_cells_ids].flatten())[1:]
    border_cells_ids = np.unique(np.r_[border_cells_ids, border_cells_neighbors])

border_cells_ids = set(border_cells_ids)
point_ids_to_exclude = []
for point_id in range(len(points)):
    cell_in_grid = locator.FindCell(points[point_id])
    if cell_in_grid < 0 or cell_in_grid in border_cells_ids:
        point_ids_to_exclude.append(point_id)
point_ids_to_exclude = set(point_ids_to_exclude)
print('len(point_ids_to_exclude):', len(point_ids_to_exclude))

facet_ids_to_exclude = []
for facet_id in range(len(facets)):
    for point_id in facets[facet_id]:
        if point_id in point_ids_to_exclude:
            facet_ids_to_exclude.append(facet_id)
facet_ids_to_exclude = np.unique(np.array(facet_ids_to_exclude))
print('len(facet_ids_to_exclude):', len(facet_ids_to_exclude))

new_facets = np.delete(facets, facet_ids_to_exclude, axis=0)
new_points, new_facets = remove_unused_points(points, new_facets)
print(new_points.shape, new_facets.shape)

writeVtkGrid(constructVtkGrid(new_points, new_facets), 'grids/corrected_inner_surfaces.vtk')


