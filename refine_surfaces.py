from utils import *


def out_res(_points, _facets, filename='grids/tmp.vtk'):
    _min_height, _asp_ratio = surface_quality(_points, _facets, draw_picture=True)
    vtk_grid = constructVtkGrid(_points, _facets,
                                cell_info={
                                    'log10_min_h': ('float', np.log10(_min_height)),
                                    'asp_ratio': ('float', _asp_ratio)
                                })
    writeVtkGrid(vtk_grid, filename)


def basic_cleaning(points, facets):
    facets = np.sort(facets, axis=1)

    facets = clear_from_self_intersections(facets)

    facets = replace_small_facets_and_its_neighbors(points, facets)
    facets = clear_from_self_intersections(facets)

    facets = choose_only_big_connected_components(facets)
    points, facets = remove_unused_points(points, facets)

    facets = orient_facets(facets)
    assert np.all(facets == orient_facets(facets))
    assert np.all(facets == clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets))
    assert np.all(facets == clear_from_cases_when_more_than_one_surface_region_contain_one_point(facets))
    print(np.histogram([len(n) for e, n in construct_edges(facets).items()], 6, (0, 6)))
    return points, facets


# points, facets, _ = convertVtkGridToNumpy(readVtkGrid("grids/initial/surfaces_initial.vtk"))
# points, facets = basic_cleaning(points, facets)

# points, facets, _ = convertVtkGridToNumpy(readVtkGrid("grids/surfaces_basically_cleared.vtk"))
# facets = np.sort(facets, axis=1)

# out_res(points, facets)
# meshio.write_points_cells("grids/skull.off", points, {'triangle': facets})
# remove_comments("grids/skull.off")

# fname = "skull_filled_remeshed"
# to_vtk('grids/{}.off'.format(fname))
# points, facets, _ = convertVtkGridToNumpy(readVtkGrid("grids/{}.vtk".format(fname)))
# print(surface_quality(points, facets, True))

# min_height, asp_ratio = surface_quality(points, facets)
# meshio.write_points_cells(
#     "grids/test_meshio_write.vtk", points, {'triangle': facets},
#     cell_data={'triangle': {'min_h': min_height}})

# ps, cs, _ = convertVtkGridToNumpy(readVtkGrid("grids/skull.1.vtk"))
# grid_quality(ps, cs)

points, facets, _ = convertVtkGridToNumpy(readVtkGrid("grids/inner_surfaces.vtk"))
facets = np.sort(facets, axis=1)

facets = replace_small_facets_and_its_neighbors(points, facets)
facets = clear_from_self_intersections(facets)
facets = choose_only_big_connected_components(facets)

points, facets = remove_unused_points(points, facets)
assert np.all(facets == clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets))
assert np.all(facets == clear_from_cases_when_more_than_one_surface_region_contain_one_point(facets))
print(np.histogram([len(n) for e, n in construct_edges(facets).items()], 6, (0, 6)))





