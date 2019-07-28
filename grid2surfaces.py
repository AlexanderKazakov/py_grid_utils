from utils import *


# points, cells = readInmGrid("grids/mesh-coarse.out")
points, cells, materials = convertVtkGridToNumpy(readVtkGrid('grids/mesh_initial.vtk'))
cells = np.c_[cells, materials]
bound_facets = findBoundFacets(cells)


def construct_surfaces_simple(bound_facets):
    # remove duplicate facets for different materials
    different_material = np.where(np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1))[0]
    bound_facets = np.delete(bound_facets, different_material, axis=0)
    bound_facets = bound_facets[:, :-1]

    surface_grid = constructVtkGrid(points, bound_facets)
    writeVtkGrid(surface_grid, "grids/surfaces.vtk")


# convert bound facets: for each facet two materials which it divides
diff_materials_second_id = np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1)
diff_materials = np.logical_or(np.append(diff_materials_second_id, False), np.insert(diff_materials_second_id, 0, False))
bound_facets_h = np.insert(bound_facets, np.where(np.logical_not(diff_materials))[0], np.array([-1] * 4, dtype=np.int64), axis=0)
bound_facets = np.c_[bound_facets_h[1::2], bound_facets_h[::2, -1]]
bound_facets = np.c_[bound_facets[:, :-2], np.sort(bound_facets[:, -2:], axis=1)]

unique_materials = np.unique(bound_facets[:, -2:])
for cntr1 in range(len(unique_materials)):
    for cntr2 in range(cntr1 + 1, len(unique_materials)):
        m1 = unique_materials[cntr1]
        m2 = unique_materials[cntr2]
        assert m1 < m2
        surface_facets = bound_facets[np.logical_and(bound_facets[:, -2] == m1, bound_facets[:, -1] == m2), :-2]
        if surface_facets.size == 0:
            continue
        surface_points, surface_facets = remove_unused_points(points, surface_facets)
        print(m1, m2)
        min_h, asp_ratio = surface_quality(surface_points, surface_facets)
        surface_grid = constructVtkGrid(surface_points, surface_facets, cell_info={
            'log10_min_h': ('float', np.log10(min_h)), 'asp_ratio': ('float', asp_ratio)})
        writeVtkGrid(surface_grid, 'grids/split_surfaces/{}_{}.vtk'.format(m1, m2))









