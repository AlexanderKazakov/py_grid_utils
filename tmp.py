from utils import *


fname = 'grids/results/9_percent/skull.vtk'
# fname = 'grids/results/grid_with_materials_from_naively_refined_surfaces_004.vtk'
# fname = 'grids/initial/cube.1.vtk'
vtk_grid = meshio.read(fname)
points = vtk_grid.points
cells = vtk_grid.cells['tetra']
materials = vtk_grid.cell_data['tetra']['medit:ref']

points /= points.max(axis=0) - points.min(axis=0)
cp = points[cells]
cell_centers = cp.sum(axis=1) / 4
part8 = np.all(cell_centers > points.mean(axis=0), axis=1)
cells = cells[part8]
materials = materials[part8]
points, cells = remove_unused_points(points, cells)
meshio.write_points_cells(
    'grids/skull8.vtk', points, {'tetra': cells},
    cell_data={'tetra': {'material': materials}}
)

























