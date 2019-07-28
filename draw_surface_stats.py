from utils import *


def add_cell(filename='grids/cells.csv'):
    global cells
    data = np.recfromcsv(filename)
    point_ids = data['point_id']
    cells = np.concatenate([cells, point_ids.reshape(1, 3)])


def add_cells_round(filename='grids/points.csv'):
    global points, cells
    data = np.recfromcsv(filename)
    point_ids = data['point_id']
    point_coords = points[point_ids]
    new_point_coords = np.mean(point_coords, axis=0)
    points = np.concatenate([points, new_point_coords.reshape(1, 3)])
    new_point_id = points.shape[0] - 1
    add_cells = []
    for p1, p2 in zip(point_ids, np.roll(point_ids, -1)):
        add_cells.append([new_point_id, p1, p2])
    add_cells = np.array(add_cells)
    cells = np.concatenate([cells, add_cells])


def delete_cells(filename='grids/cells.csv'):
    global cells
    data = np.recfromcsv(filename)
    cell_ids = data['cell_id']
    cells = np.delete(cells, cell_ids, axis=0)


def add_patch(filename='grids/patch_remeshed.vtk'):
    global points, cells
    patch_points, patch_cells = convertVtkGridToNumpy(readVtkGrid(filename))
    for p_local_id in range(len(patch_points)):
        p = patch_points[p_local_id]
        p_global_ids = np.where(np.all(points == p, axis=1))[0]
        assert 0 <= len(p_global_ids) <= 1
        if len(p_global_ids) == 0:
            points = np.concatenate([points, patch_points[p_local_id].reshape(1, 3)])
            p_global_id = points.shape[0] - 1
        else:
            p_global_id = p_global_ids[0]
        print(p_local_id, '->', p_global_id)
        patch_cells[patch_cells == p_local_id] = p_global_id
    cells = np.concatenate([cells, patch_cells])


def out_res(filename='grids/tmp.vtk'):
    min_h = surface_quality(points, cells)
    vtk_grid = constructVtkGrid(
        points, cells,
        cell_info={
            'log10_min_h': ('float', np.log10(min_h)),
            # 'cell_id': ('int', np.arange(cells.shape[0]))
        },
        point_info={
            # 'point_id': ('int', np.arange(points.shape[0]))
        }
    )
    writeVtkGrid(vtk_grid, filename)


points, cells = convertVtkGridToNumpy(readVtkGrid("grids/surfaces.vtk"))
out_res('grids/surfaces_with_stats.vtk')

# delete_cells()
# add_patch()
# surface_quality(points, cells)
# writeVtkGrid(constructVtkGrid(points, cells), 'grids/surfaces.vtk')

# min_h = surface_quality(points, cells)
# cells = cells[np.log10(min_h) > -0.2]
# writeVtkGrid(constructVtkGrid(points, cells), 'grids/surfaces.vtk')



