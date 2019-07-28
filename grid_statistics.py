from utils import *


# best found params for cube with side 100 (min / median = 0.195):
# tetgen.exe -p -q2/15 -a1.0 -k -V "C:\DATA\Programs\tetgen\cube.poly"

# points, cells = readInmGrid("grids/mesh-coarse.out")
# materials = cells[:, -1]
# cells = cells[:, :-1]
# for m in np.unique(materials):
#     plotMinHeightHistogram(points, cells[materials == m])

# vtk_grid = readVtkGrid("grids/result.vtk")
# points, cells, materials = convertVtkGridToNumpy(vtk_grid)
# plotMinHeightHistogram(points, cells[materials[:, 0] != 0])

# fname = r"grids/mesh_initial.vtk"
# fname = r"C:\DATA\Programs\SALOME-9.2.2\SAMPLES\Mesh_1.vtk"
# fname = r"C:\DATA\Programs\SALOME-9.2.2\SAMPLES\Bone_mesh.vtk"
fname = r"grids/surfaces.1.vtk"
vtk_grid = readVtkGrid(fname)
points, cells = convertVtkGridToNumpy(vtk_grid)
grid_quality(points, cells)


# vtk_grid = readVtkGrid(r"grids/mesh_initial.vtk")
# points, cells, materials = convertVtkGridToNumpy(vtk_grid)
# bound_facets = findBoundFacets(np.c_[cells, materials])
# surface_quality(points, bound_facets[:, :-1])

# !!! 0.043 h -- C:\DATA\Programs\tetgen\build_dir\Release\tetgen.exe -p -q2/15 -Y -k -V "C:\Users\alex\PycharmProjects\untitled\grids\surfaces.stl"




