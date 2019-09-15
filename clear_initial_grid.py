from utils import *

points, cells, materials = convertVtkGridToNumpy(readVtkGrid('grids/initial/mesh_initial.vtk'))
materials = materials[0]
materials = np.append(materials, 6)
facets = construct_facets_from_cells(cells)
neighbors = find_cells_neighbors(cells)

def find_connected_components():
    graph = nx.empty_graph()
    for facet, cell_ids in facets.items():
        assert 1 <= len(cell_ids) <= 2
        if len(cell_ids) == 2:
            graph.add_node(cell_ids[0])
            graph.add_node(cell_ids[1])
            if materials[cell_ids[0]] == materials[cell_ids[1]]:
                graph.add_edge(cell_ids[0], cell_ids[1])
        else:
            graph.add_node(cell_ids[0])
    print('Edges num:', graph.number_of_edges())
    conn_comps = list(nx.connected_component_subgraphs(graph))
    print('Connected components num:', len(conn_comps))
    number_of_cells = [cc.number_of_nodes() for cc in conn_comps]
    # plt.hist(np.log10(number_of_cells), 100)
    # plt.show()
    return conn_comps, number_of_cells

while True:
    conn_comps, number_of_cells = find_connected_components()
    old_ms = materials.copy()
    for i in range(len(conn_comps)):
        if number_of_cells[i] < 1000:
            conn_cells_ids = np.array([n for n in conn_comps[i].nodes()], dtype=np.int64)
            nbrs = np.unique(neighbors[conn_cells_ids].flatten())
            nbrs_ms = materials[nbrs]
            curr_mat = np.unique(materials[conn_cells_ids])
            assert len(curr_mat) == 1
            curr_mat = curr_mat[0]
            nbrs_ms = nbrs_ms[nbrs_ms != curr_mat]
            mat_hist = np.histogram(nbrs_ms, np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))[0]
            if np.any(mat_hist != 0):
                new_mat = np.argmax(mat_hist)
                materials[conn_cells_ids] = new_mat
                print(i, ':', conn_cells_ids)
                print(mat_hist)
                print(curr_mat, '-->', new_mat, '\n==============================================')
    changed_materials_num = np.sum(old_ms != materials)
    print('Changed materials num:', changed_materials_num)
    if changed_materials_num == 0:
        for i in range(len(conn_comps)):
            if number_of_cells[i] < 1000:
                conn_cells_ids = np.array([n for n in conn_comps[i].nodes()], dtype=np.int64)
                assert np.all(materials[conn_cells_ids] == 6)
        break

materials = materials[:-1]
cells = cells[materials != 6]
materials = materials[materials != 6]
points, cells = remove_unused_points(points, cells)

writeVtkGrid(constructVtkGrid(points, cells, {'material': ('int', materials)}), 'grids/big_conn_comps.vtk')
