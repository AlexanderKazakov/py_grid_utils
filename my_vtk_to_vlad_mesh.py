from utils import *

# cube:
vtk_grid = meshio.read('grids/initial/cube.1.vtk')
points = vtk_grid.points
cells = vtk_grid.cells['tetra']
materials = np.zeros(cells.shape[0], np.uint8)
out_name = 'cube'
r = 0.25
r0 = np.array([0.5, 0.5, 0])

# skull8:
# vtk_grid = meshio.read('grids/skull8.vtk')
# points = vtk_grid.points
# cells = vtk_grid.cells['tetra']
# materials = vtk_grid.cell_data['tetra']['material']
# out_name = 'skull8'
# r = 0.12
# r0 = np.array([0.8, 0.7, 0.75])

# skull:
# vtk_grid = meshio.read('grids/results/9_percent/skull.vtk')
# points = vtk_grid.points
# cells = vtk_grid.cells['tetra']
# materials = vtk_grid.cell_data['tetra']['material']
# out_name = 'skull'
# r = 0.12
# r0 = np.array([0.6, 0.525, 0.7])

print('AABB:', points.min(axis=0), points.max(axis=0), sep='\n')

border_facets = find_border_facets(np.c_[cells, materials])[:, :-1]

# sort border facets in order to make ones with similar border condition contiguous
bf_centers = points[border_facets].sum(axis=1) / 3

# in sphere
# bf_in_area = np.sum((bf_centers - r0) ** 2, axis=1) < r ** 2

# in rectangle (for cube)
bf_in_area = np.logical_and(np.abs(bf_centers[:, 2]) < 1e-14,
                            np.logical_and(0.2 < bf_centers[:, 1], bf_centers[:, 1] < 0.8))

border_facets = border_facets[np.argsort(bf_in_area)]
num_bf_cond_1 = np.sum(bf_in_area)
num_bf_cond_0 = bf_in_area.size - num_bf_cond_1

# write to Vlad's mesh (see MeshIO<Space3>::LoadBoundaries)
with open('grids/results/for_dgm/{}.mesh'.format(out_name), 'w') as f:
    # points
    f.write(str(points.shape[0]) + '\n')
    for p in points:
        f.write('{} {} {}\n'.format(p[0], p[1], p[2]))
    f.write('\n')

    # cells
    f.write(str(cells.shape[0] * 4) + '\n')  # note 4 !
    for c in cells:
        f.write('{} {} {} {}\n'.format(c[0], c[1], c[2], c[3]))
    f.write('\n')

    # contacts
    f.write('0 0\n')
    f.write('\n')

    # borders
    f.write(str(border_facets.shape[0]) + '\n')
    for bf in border_facets:
        f.write('{} {} {}\n'.format(bf[0], bf[1], bf[2]))
    # number of different boundary conditions types,
    # then number of faces with that condition (faces with the same condition go contiguously)
    #
    # for one boundary condition for all cells:
    # f.write('1 {}\n'.format(border_facets.shape[0]))
    # f.write('\n')
    #
    # for two boundary conditions:
    f.write('2 {} {}\n'.format(num_bf_cond_0, num_bf_cond_1))
    f.write('\n')

    # detectors
    f.write('0\n')

materials = materials.astype(np.uint8)
with open('grids/results/for_dgm/{}.params'.format(out_name), 'wb') as f:
    f.write(materials.tostring())


