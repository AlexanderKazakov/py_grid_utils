import numpy as np
import re

with open('grids/results/grid_with_materials.vtk', 'r') as f:
    lines = f.readlines()

for line_cntr in range(len(lines)):
    match = re.search(r'\s*POINTS\s+(\d+)\s+double\s*', lines[line_cntr])
    if match is not None:
        num_points = int(match.group(1))
        break
print('Number of points:', num_points)

points = []
for line_cntr in range(line_cntr + 1, len(lines)):
    match = re.search(r'\s*CELLS\s+(\d+)\s+\d+\s*', lines[line_cntr])
    if match is None:
        points.extend([float(v) for v in lines[line_cntr].split()])
    else:
        num_cells = int(match.group(1))
        break

points = np.array(points).reshape(-1, 3)
assert points.shape[0] == num_points
print('Number of cells:', num_cells)

cells = []
for line_cntr in range(line_cntr + 1, len(lines)):
    match = re.search(r'\s*CELL_TYPES\s+(\d+)\s*', lines[line_cntr])
    if match is None:
        cells.extend([int(v) for v in lines[line_cntr].split()][1:])
    else:
        num_cells_types = int(match.group(1))
        break

cells = np.array(cells, dtype=np.int64).reshape(-1, 4)
assert cells.shape[0] == num_cells == num_cells_types

for line_cntr in range(line_cntr + 1, len(lines)):
    match = re.search(r'\s*cell_info*', lines[line_cntr])
    if match is not None:
        break

cell_ids = []
for line_cntr in range(line_cntr + 1, len(lines)):
    cell_ids.extend([int(v) for v in lines[line_cntr].split()])

cell_ids = np.array(cell_ids, dtype=np.uint8)
assert cell_ids.shape[0] == num_cells


# center to zero
points = points - points.mean(axis=0)
print('AABB:', points.min(axis=0), points.max(axis=0), sep='\n')


# write to Vlad's mesh
with open('grids/results/skull.mesh', 'w') as f:
    f.write(str(num_points) + '\n')
    for p in points:
        f.write('{} {} {}\n'.format(p[0], p[1], p[2]))

    f.write('\n')

    f.write(str(num_cells * 4) + '\n')  # note 4 !
    for c in cells:
        f.write('{} {} {} {}\n'.format(c[0], c[1], c[2], c[3]))

    f.write('0\n0\n\n0\n0\n0')


with open('grids/results/skull.params', 'wb') as f:
    for ci in cell_ids:
        f.write(ci)



# rho, lambda, mu, tau
# жир, мышцы, мозг, кости, сосуды
# 0.916,  1.415, 0.236, 1.585
# 1.041,  1.968, 0.331, 0.878
# 1.030,  1.856, 0.309, 1.293
# 1.904,  5.891, 0.982, 0.000
# 1.066,  2.088, 0.348, 1.288


# <Submesh index="0" rho="0.916" lambda="1.415" mju="0.236" />
# <Submesh index="1" rho="0.916" lambda="1.415" mju="0.236" />
# <Submesh index="2" rho="1.041" lambda="1.968" mju="0.331" />
# <Submesh index="3" rho="1.030" lambda="1.856" mju="0.309" />
# <Submesh index="4" rho="1.904" lambda="5.891" mju="0.982" />
# <Submesh index="5" rho="1.066" lambda="2.088" mju="0.348" />




