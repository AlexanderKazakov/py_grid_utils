from utils import *


src_points, src_cells = readInmGrid("grids/mesh-coarse.out")
vtk_grid = constructVtkGridFromCells(
    src_points, src_cells[:, :-1], src_cells[:, -1])
writeVtkGrid(vtk_grid, "grids/mesh_initial.vtk")






