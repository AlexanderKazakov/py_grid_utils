from utils import *


fname = 'grids/results/9_percent/skull.vtk'
# fname = 'grids/results/grid_with_materials_from_naively_refined_surfaces_004.vtk'
# fname = 'grids/initial/cube.1.vtk'
# fname = 'grids/skull8.vtk'
vtk_grid = meshio.read(fname)
points = vtk_grid.points
cells = vtk_grid.cells['tetra']
# materials = vtk_grid.cell_data['tetra']['material']

min_height, asp_ratio = grid_quality(points, cells)
ac_vel = 4
p = 2
print('tau = ', min_height.min() / ac_vel / (2 * p + 1))

# writeVtkGrid(constructVtkGrid(points, cells, {
#     'material': ('int', materials),
#     'log10_min_h': ('float', np.log10(min_height)),
#     'asp_ratio': ('float', asp_ratio),
# }), 'grids/res.vtk')

# bound_facets = findBoundFacets(np.c_[cells, materials])
# different_material = np.where(np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1))[0]
# bound_facets = np.delete(bound_facets, different_material, axis=0)
# bound_facets = bound_facets[:, :-1]
# new_points, new_facets = remove_unused_points(points, bound_facets)
# surface_grid = constructVtkGrid(points, bound_facets)
# writeVtkGrid(surface_grid, "grids/surfaces.vtk")
# to_stl('grids/surfaces.vtk')

# for i in range(15):
#     filename = 'grids/out_saved/snapshot[000]0000{:02}.vti'.format(i)
#     reader = vtk.vtkXMLImageDataReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     imageData = reader.GetOutput()
#     print(imageData.GetPointData().GetArray(0).GetRange())
#     print(imageData.GetPointData().GetArray(1).GetRange())
#     print('---')


# vtk_grid = meshio.read('grids/results/9_percent/res.vtk')
# vtk_grid = meshio.read('grids/initial/mesh_initial.vtk')
# points = vtk_grid.points / 1000
# cells = vtk_grid.cells['tetra']
# materials = vtk_grid.cell_data['tetra']['material']
# materials = vtk_grid.cell_data['tetra']['cell_info']
# min_height, asp_ratio = grid_quality(points, cells)

# params = {
#     1: {'rho':  916, 'lambda': 1.415e+9, 'mu': 0.236e+9, 'tau': 1.585e-5},
#     2: {'rho': 1041, 'lambda': 1.968e+9, 'mu': 0.331e+9, 'tau': 0.878e-5},
#     3: {'rho': 1030, 'lambda': 1.856e+9, 'mu': 0.309e+9, 'tau': 1.293e-5},
#     4: {'rho': 1904, 'lambda': 5.891e+9, 'mu': 0.982e+9, 'tau': 0.000e-5},
#     5: {'rho': 1066, 'lambda': 2.088e+9, 'mu': 0.348e+9, 'tau': 1.288e-5},
# }
#
# rho = np.array([params[m]['rho'] for m in materials])
# lamda = np.array([params[m]['lambda'] for m in materials])
# mu = np.array([params[m]['mu'] for m in materials])
#
# c = np.sqrt((lamda + 2 * mu) / rho)
# tau = min_height / c
# print(tau.min(), np.median(tau))
# plt.hist(tau, 1000)

