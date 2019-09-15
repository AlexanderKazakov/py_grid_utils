from utils import *


# unused
def construct_surfaces_simple(bound_facets):
    # remove duplicate facets for different materials
    different_material = np.where(np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1))[0]
    bound_facets = np.delete(bound_facets, different_material, axis=0)
    bound_facets = bound_facets[:, :-1]

    surface_grid = constructVtkGrid(points, bound_facets)
    writeVtkGrid(surface_grid, "grids/surfaces.vtk")


points, cells, materials = convertVtkGridToNumpy(readVtkGrid('grids/big_conn_comps.vtk'))
materials = materials[0]
cells = np.c_[cells, materials]
bound_facets = findBoundFacets(cells)

# convert bound facets: for each bound facet two materials which it divides
diff_materials_second_id = np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1)
diff_materials = np.logical_or(np.append(diff_materials_second_id, False), np.insert(diff_materials_second_id, 0, False))
bound_facets_h = np.insert(bound_facets, np.where(np.logical_not(diff_materials))[0], np.array([-1] * 4, dtype=np.int64), axis=0)
bound_facets = np.c_[bound_facets_h[1::2], bound_facets_h[::2, -1]]
bound_facets = np.c_[bound_facets[:, :-2], np.sort(bound_facets[:, -2:], axis=1)]

# remove vessels facets as too small (TODO)
bound_facets = bound_facets[bound_facets[:, -1] != 5]
# remove small triangles
min_height, _ = surface_quality(points, bound_facets[:, :3])
bound_facets = bound_facets[min_height > 1.5]

edges = construct_edges(bound_facets[:, :3])
facet_ids_to_rm = set()
for edge, facet_ids in edges.items():
    if len(facet_ids) != 2:  # remove multi-intersections and corners
        for f_id in facet_ids:
            facet_ids_to_rm.add(f_id)

bound_facets = np.delete(bound_facets, facet_ids, axis=0)
edges = construct_edges(bound_facets[:, :3])

graph = nx.empty_graph()
for edge, facet_ids in edges.items():
    if len(facet_ids) != 2:
        continue  # remove multi-intersections and corners
    m1 = bound_facets[facet_ids[0], -2:]
    m2 = bound_facets[facet_ids[1], -2:]
    if np.all(m1 == m2) and np.any(m1 != np.array([-1, 1])):  # remove outer surface
        graph.add_edge(facet_ids[0], facet_ids[1])

print('Edges num:', graph.number_of_edges())
conn_comps = list(nx.connected_component_subgraphs(graph))
conn_comps = [cc for cc in conn_comps if cc.number_of_nodes() > 100]
print('Connected components num:', len(conn_comps))
# plt.hist(np.log10([cc.number_of_nodes() for cc in conn_comps]), 100)
# plt.show()

new_bounds = []
for cc in conn_comps:
    new_bounds.extend(n for n in cc.nodes())
new_bounds = np.array(new_bounds)
res_facets = bound_facets[new_bounds]
res_facets[res_facets == -1] = 0
res_ms = 10 * res_facets[:, -2] + res_facets[:, -1]
vtk_grid = constructVtkGrid(points, res_facets[:, :3], {'ms': ('int', res_ms)})
writeVtkGrid(vtk_grid, 'grids/tmp.vtk')




# unique_materials = np.unique(bound_facets[:, -2:])
# for cntr1 in range(len(unique_materials)):
#     for cntr2 in range(cntr1 + 1, len(unique_materials)):
#         m1 = unique_materials[cntr1]
#         m2 = unique_materials[cntr2]
#         assert m1 < m2
#         surface_facets = bound_facets[np.logical_and(bound_facets[:, -2] == m1, bound_facets[:, -1] == m2), :-2]
#         if surface_facets.size == 0:
#             continue
#         surface_points, surface_facets = remove_unused_points(points, surface_facets)
#         print(m1, m2)
#         min_h, asp_ratio = surface_quality(surface_points, surface_facets)
#         surface_grid = constructVtkGrid(surface_points, surface_facets, cell_info={
#             'log10_min_h': ('float', np.log10(min_h)), 'asp_ratio': ('float', asp_ratio)})
#         writeVtkGrid(surface_grid, 'grids/split_surfaces/{}_{}.vtk'.format(m1, m2))









