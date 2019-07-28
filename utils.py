import numpy as np
import vtk
import matplotlib.pyplot as plt
import meshio
import networkx as nx


def constructVtkGrid(points, cells, cell_info=None, point_info=None):
    grid = vtk.vtkUnstructuredGrid()
    grid.Allocate(cells.shape[0])
    cell_size = cells.shape[1]
    if cell_size == 4:
        cell_type = vtk.vtkTetra
    elif cell_size == 3:
        cell_type = vtk.VTK_TRIANGLE
    else:
        raise ValueError('Unknown cell type')
    for cell in cells:
        grid.InsertNextCell(cell_type, cell_size, cell)

    vtk_points = vtk.vtkPoints()
    for i, point in enumerate(points):
        vtk_points.InsertPoint(i, point)
    grid.SetPoints(vtk_points)

    def construct_vtk_arr(name, data_type, data):
        if data_type == 'int':
            arr_vtk = vtk.vtkIntArray()
        elif data_type == 'float':
            arr_vtk = vtk.vtkFloatArray()
        else:
            raise ValueError('Unknown data type')
        arr_vtk.SetName(name)
        arr_vtk.SetNumberOfComponents(1)
        arr_vtk.Allocate(data.shape[0])
        for cnt in range(data.shape[0]):
            arr_vtk.InsertNextValue(data[cnt])
        return arr_vtk

    if cell_info is not None:
        for name, (data_type, data) in cell_info.items():
            grid.GetCellData().AddArray(construct_vtk_arr(name, data_type, data))
    if point_info is not None:
        for name, (data_type, data) in point_info.items():
            grid.GetPointData().AddArray(construct_vtk_arr(name, data_type, data))

    return grid


def writeVtkGrid(grid, filename):
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()


def readVtkGrid(filename):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def showVtkGrid(ugrid):
    colors = vtk.vtkNamedColors()

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ugridMapper = vtk.vtkDataSetMapper()
    ugridMapper.SetInputData(ugrid)

    ugridActor = vtk.vtkActor()
    ugridActor.SetMapper(ugridMapper)
    ugridActor.GetProperty().SetColor(colors.GetColor3d("Peacock"))
    # ugridActor.GetProperty().SetAmbient(0.5)
    ugridActor.GetProperty().SetOpacity(0.5)
    # ugridActor.GetProperty().SetDiffuse(0.5)
    ugridActor.GetProperty().EdgeVisibilityOn()
    renderer.AddActor(ugridActor)

    # оси координат
    # renderer.AddActor(vtk.vtkAxesActor())

    renderer.SetBackground(colors.GetColor3d("Beige"))
    renderer.ResetCamera()
    renderer.GetActiveCamera().SetPosition(0, 0, 10)
    renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
    # renderer.GetActiveCamera().Elevation(0.0)
    # renderer.GetActiveCamera().Azimuth(0.0)
    # renderer.GetActiveCamera().Dolly(1.2)

    renWin.SetSize(1000, 1000)
    renWin.Render()
    iren.Start()


def convertVtkGridToNumpy(grid):
    num_cells = grid.GetNumberOfCells()
    num_points = grid.GetNumberOfPoints()
    points = np.zeros((num_points, 3), dtype=np.float64)
    for cnt in range(num_points):
        points[cnt, :] = grid.GetPoint(cnt)
    n_points_in_cell = grid.GetCell(0).GetNumberOfPoints()
    print('n_points_in_cell:', n_points_in_cell)

    cells = np.zeros((num_cells, n_points_in_cell), dtype=np.int64)
    for cnt in range(num_cells):
        assert grid.GetCell(cnt).GetNumberOfPoints() == n_points_in_cell
        if grid.GetCell(cnt).GetNumberOfPoints() == n_points_in_cell:
            cells[cnt, :] = [grid.GetCell(cnt).GetPointId(i) for i in range(n_points_in_cell)]
    cells = np.delete(cells, np.where(np.all(cells == 0, axis=1))[0], axis=0)

    assert grid.GetPointData().GetNumberOfArrays() == 0, 'TODO'
    cell_info_arrs = []
    for i in range(grid.GetCellData().GetNumberOfArrays()):
        arr = grid.GetCellData().GetArray(i)
        if isinstance(arr, vtk.vtkFloatArray):
            dtype = np.float
        elif isinstance(arr, vtk.vtkIntArray):
            dtype = np.int64
        else:
            raise RuntimeError('Unknown array data type, TODO')
        data = np.zeros((num_cells, arr.GetNumberOfComponents()), dtype=dtype)
        for cnt in range(num_cells):
            data[cnt] = arr.GetValue(cnt)
        cell_info_arrs.append(data)
    return points, cells, cell_info_arrs


def readInmGrid(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = int(lines[0])
    num_cells = int(lines[num_points + 1])

    points_lines = lines[1:num_points + 1]
    cells_lines = lines[num_points + 2:num_points + 2 + num_cells]

    points = np.array([[float(val) for val in point.split(' ')] for point in points_lines], dtype=np.float64)
    cells = np.array([[int(val) for val in cell.split(' ')] for cell in cells_lines])
    cells[:, :-1] = cells[:, :-1] - 1

    return points, cells


def findBoundFacets(cells):
    facets = np.vstack([cells[:, 1:],
                        np.hstack([cells[:, :1], cells[:, 2:]]),
                        np.hstack([cells[:, :2], cells[:, 3:]]),
                        np.hstack([cells[:, :3], cells[:, 4:]])])
    # sort vertices id in each facet
    facets = np.hstack([np.sort(facets[:, :-1], axis=1), facets[:, -1:]])
    # sort all facets for fast search
    facets = np.rot90(np.rot90(facets)[:, np.lexsort(np.rot90(facets))], k=-1)
    # find facets which do not bound materials
    same_material = np.where(np.all(np.diff(facets, axis=0) == 0, axis=1))[0]
    bound_facets = np.delete(facets, np.c_[same_material, same_material + 1], axis=0)
    # split border from contact facets
    # different_material = np.where(np.all(np.diff(bound_facets[:, :-1], axis=0) == 0, axis=1))[0]
    # contact_facets = bound_facets[np.c_[different_material, different_material + 1]]
    # border_facets = np.delete(bound_facets, np.c_[different_material, different_material + 1], axis=0)
    return bound_facets


def grid_quality(points, cells):
    c = points[cells]
    r = np.stack([c[:, 1, :] - c[:, 0, :], c[:, 2, :] - c[:, 0, :], c[:, 3, :] - c[:, 0, :]], axis=1)
    vol = np.abs(np.linalg.det(r)) / 6
    areas = []
    for cnt in range(4):
        i, j, k = tuple(np.delete(np.arange(4, dtype=np.int64), cnt))
        areas.append(np.linalg.norm(np.cross(c[:, i, :] - c[:, k, :], c[:, j, :] - c[:, k, :]), axis=1) / 2)
    max_side_area = np.vstack(areas).max(axis=0)
    minimal_height = 3 * vol / max_side_area
    print('min / median (minimal height) =', np.min(minimal_height), '/', np.median(minimal_height), '=',
          np.min(minimal_height) / np.median(minimal_height))

    longest_edge = []
    for i in range(4):
        for j in range(i):
            longest_edge.append(np.linalg.norm(c[:, i, :] - c[:, j, :], axis=1))
    longest_edge = np.vstack(longest_edge).max(axis=0)
    print('max / median (longest edge) =',  np.max(longest_edge), '/', np.median(longest_edge), '=',
          np.max(longest_edge) / np.median(longest_edge))

    asp_ratio = longest_edge / minimal_height
    print('max / median (aspect ratio) =', np.max(asp_ratio), '/', np.median(asp_ratio), '=',
          np.max(asp_ratio) / np.median(asp_ratio))

    plt.scatter(np.log10(minimal_height), asp_ratio)
    plt.title("Quality measures correlation of 3D tetrahedral grid")
    plt.xlabel("log10(minimal height)")
    plt.ylabel("aspect ratio")
    plt.grid(True)
    plt.show()


def surface_quality(points, cells, draw_picture=False):
    c = points[cells]
    r = np.stack([c[:, 1, :] - c[:, 0, :], c[:, 2, :] - c[:, 0, :]], axis=1)
    area = np.linalg.norm(np.cross(r[:, 0, :], r[:, 1, :]), axis=1) / 2
    side = []
    for cnt in range(3):
        i, j = tuple(np.delete(np.arange(3, dtype=np.int64), cnt))
        side.append(np.linalg.norm(c[:, i, :] - c[:, j, :], axis=1))
    longest_edge = np.vstack(side).max(axis=0)
    minimal_height = 2 * area / longest_edge
    print('min / median (min height) =', np.min(minimal_height) / np.median(minimal_height))

    asp_ratio = longest_edge / minimal_height
    print('max / median (asp ratio) =', np.max(asp_ratio) / np.median(asp_ratio))

    if draw_picture:
        plt.scatter(np.log10(minimal_height), asp_ratio)
        plt.title("Quality measures correlation of 2D triangular surface")
        plt.xlabel("log10(minimal height)")
        plt.ylabel("aspect ratio")
        plt.grid(True)
        plt.show()

    return minimal_height, asp_ratio


def remove_unused_points(points, cells):
    cells_shape = cells.shape
    cells = cells.flatten()
    used_points = np.unique(cells)
    points = points[used_points]
    transform = {old: new for new, old in enumerate(used_points)}
    for i in range(len(cells)):
        cells[i] = transform[cells[i]]
    cells = cells.reshape(cells_shape)
    return points, cells


def construct_facets_from_cells(cells):
    cells = np.sort(cells, axis=1)
    facets = {}
    for counter, cell in enumerate(cells):
        for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            facet = cell[i], cell[j], cell[k]
            if facets.get(facet) is None:
                facets[facet] = [counter]
            else:
                facets[facet] += [counter]
    return facets


def find_cells_neighbors(cells):
    facets = construct_facets_from_cells(cells)
    neighbors_dict = {i: [] for i in range(len(cells))}
    for facet, cell_ids in facets.items():
        if len(cell_ids) == 1:
            a = cell_ids[0]
            neighbors_dict[a].append(-1)
        elif len(cell_ids) == 2:
            a, b = cell_ids[0], cell_ids[1]
            neighbors_dict[a].append(b)
            neighbors_dict[b].append(a)
        else:
            raise RuntimeError('Unexpected number of cells in contact')
    res = -2 * np.ones((len(cells), 4), dtype=np.int64)
    for i, neighbors in neighbors_dict.items():
        assert len(neighbors) == 4
        res[i] = np.sort(neighbors)
    assert np.all(res != -2)
    return res


def construct_edges(_facets):
    facets = np.sort(_facets, axis=1)
    edges = {}
    for counter, face in enumerate(facets):
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            edge = face[i], face[j]
            if edges.get(edge) is None:
                edges[edge] = [counter]
            else:
                edges[edge] += [counter]
    return edges


def find_neighbors(facets):
    edges = construct_edges(facets)
    neighbors_dict = {i: [] for i in range(len(facets))}
    for edge, face_ids in edges.items():
        if len(face_ids) == 1:
            a = face_ids[0]
            neighbors_dict[a].append(-1)
        elif len(face_ids) == 2:
            a, b = face_ids[0], face_ids[1]
            neighbors_dict[a].append(b)
            neighbors_dict[b].append(a)
        else:
            raise RuntimeError('Unexpected number of facets')
    res = -2 * np.ones((len(facets), 3), dtype=np.int64)
    for i, neighbors in neighbors_dict.items():
        assert len(neighbors) == 3
        res[i] = np.sort(neighbors)
    assert np.all(res != -2)
    return res


def orient_facets(facets_, init_facet_id=0):
    facets = facets_.copy()
    facets_neighbor_ids = find_neighbors(facets)
    already_oriented_facets_ids = set()
    last_step_oriented_facets_ids = {init_facet_id}
    while len(last_step_oriented_facets_ids) != 0:
        already_oriented_facets_ids = already_oriented_facets_ids.union(last_step_oriented_facets_ids)
        new_oriented_facets_ids = set()
        for face_id in last_step_oriented_facets_ids:
            for neighbor_id in facets_neighbor_ids[face_id]:
                if neighbor_id == -1:
                    continue  # border edge
                face = facets[face_id]
                neighbor = facets[neighbor_id]
                min_eq = min(set(face).intersection(set(neighbor)))
                face = np.roll(face, -np.where(face == min_eq)[0][0])
                neighbor = np.roll(neighbor, -np.where(neighbor == min_eq)[0][0])
                if neighbor_id not in already_oriented_facets_ids:
                    # orient neighbor
                    if np.any(face[1:] == neighbor[1:]):
                        neighbor[1:] = neighbor[2:0:-1]
                        facets[neighbor_id] = neighbor
                    # add neighbor to orient its neighbors on the next step of the cycle
                    new_oriented_facets_ids.add(neighbor_id)
                assert np.all(face[1:] != neighbor[1:])
        last_step_oriented_facets_ids = new_oriented_facets_ids
    return facets


def remove_comments(filename):
    # remove meshio comments which break CGAL genius reader
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [l for l in lines if l[0] != '#']
    with open(filename, 'w') as f:
        f.writelines(lines)


def convert_grid_to(filename_in, format):
    filename_out = '.'.join(filename_in.split('.')[:-1] + [format])
    mesh = meshio.read(filename_in)
    meshio.write(filename_out, mesh)
    if format == 'off':
        remove_comments(filename_out)


def to_vtk(filename_in):
    convert_grid_to(filename_in, 'vtk')


def to_stl(filename_in):
    convert_grid_to(filename_in, 'stl')


def to_off(filename_in):
    convert_grid_to(filename_in, 'off')


def clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets):
    edges = construct_edges(facets)
    print(np.histogram([len(edges[e]) for e in edges], 6, (0, 6)))
    bad_edges = [e for e in edges if len(edges[e]) > 2]
    bad_facets_ids = []
    for bad_edge in bad_edges:
        bad_facets_ids.extend(edges[bad_edge])
    bad_facets_ids = np.unique(np.array(bad_facets_ids))
    return np.delete(facets, bad_facets_ids, axis=0)


def clear_from_cases_when_more_than_one_surface_region_contain_one_point(facets):
    while True:
        _points = {i: [] for i in np.unique(facets.flatten())}
        for face_id, face in enumerate(facets):
            for point_id in face:
                _points[point_id].append(face_id)
        bad_facets_ids = []
        for point_id, faces_ids in _points.items():
            _edges = construct_edges(facets[faces_ids])
            _graph = nx.empty_graph()
            for _edge, _cell_ids in _edges.items():
                assert 1 <= len(_cell_ids) <= 2
                if len(_cell_ids) == 2:
                    _graph.add_edge(_cell_ids[0], _cell_ids[1])
            if len(list(nx.connected_component_subgraphs(_graph))) > 1:
                bad_facets_ids.extend(faces_ids)
        bad_facets_ids = np.unique(np.array(bad_facets_ids))
        print('len(bad_facets_ids):', len(bad_facets_ids))
        facets = np.delete(facets, bad_facets_ids, axis=0)
        if len(bad_facets_ids) == 0:
            return facets


def clear_from_self_intersections(facets):
    facets = clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets)
    assert np.all(facets == clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets))
    facets = clear_from_cases_when_more_than_one_surface_region_contain_one_point(facets)
    assert np.all(facets == clear_from_cases_when_more_than_two_facets_connected_at_one_edge(facets))
    assert np.all(facets == clear_from_cases_when_more_than_one_surface_region_contain_one_point(facets))
    return facets


def replace_small_facets_and_its_neighbors(points, facets, thresh=None):
    min_height, asp_ratio = surface_quality(points, facets)
    plt.hist(min_height, bins=100)
    plt.show()

    if thresh is None:
        print('PLEASE ENTER THRESHOLD MIN_HEIGHT VALUE:')
        thresh = float(input())

    bad_facets = np.where(min_height < thresh)[0]
    neighbors = find_neighbors(facets)
    for counter in range(2):
        bad_neighbors = neighbors[bad_facets].flatten()
        bad_facets = np.unique(np.hstack([bad_facets, bad_neighbors]))
        print('{}: bad_facets.size: {}'.format(counter, bad_facets.size))
    bad_facets = bad_facets[bad_facets >= 0]
    facets = np.delete(facets, bad_facets, axis=0)

    min_height, asp_ratio = surface_quality(points, facets, True)
    plt.hist(min_height, bins=100)
    plt.show()
    return facets


def choose_only_big_connected_components(facets):
    edges = construct_edges(facets)
    graph = nx.empty_graph()
    for edge, cell_ids in edges.items():
        assert 1 <= len(cell_ids) <= 2
        if len(cell_ids) == 2:
            graph.add_edge(cell_ids[0], cell_ids[1])
    conn_comps = list(nx.connected_component_subgraphs(graph))
    number_of_facets = [cc.number_of_nodes() for cc in conn_comps]

    plt.hist(number_of_facets, bins=100)
    plt.show()
    print('PLEASE ENTER THRESHOLD CONNECTED COMPONENTS VALUE:')
    min_number_of_facets = int(input())

    big_subgraphs = [cc for cc in conn_comps if cc.number_of_nodes() >= min_number_of_facets]
    new_facets_ids = []
    for bsg in big_subgraphs:
        new_facets_ids.extend(list(bsg.nodes()))
    new_facets_ids = np.unique(np.array(new_facets_ids))
    facets = facets[new_facets_ids]
    print(np.histogram([len(n) for e, n in construct_edges(facets).items()], 6, (0, 6)))
    return facets

