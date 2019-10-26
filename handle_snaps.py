from utils import *
import os
# from shutil import copyfile


input_folder = r'C:\DATA\dgm\.bin\out'
snaps_name = 'vtu'
output_folder = 'grids/out'
num_snaps_to_leave = 100


snaps_input = next(os.walk(input_folder))[2]
snaps_input = [s for s in snaps_input if s.find(snaps_name) == 0]
snaps_step = (len(snaps_input) - 1) // (num_snaps_to_leave - 1)
assert snaps_step > 0

for cntr in range(num_snaps_to_leave):
    fname = snaps_input[cntr * snaps_step]
    input_snap_file = os.path.join(input_folder, fname)
    output_snap_file = os.path.join(output_folder, fname)
    # copyfile(input_snap_file, output_snap_file)
    grid = meshio.read(input_snap_file)
    print('file: {}, cells: {}, points: {}, cell_data: {}'.format(
        input_snap_file, grid.cells['tetra'].shape, grid.points.shape, grid.cell_data['tetra'].keys()
    ))
    meshio.write(output_snap_file, grid)












