from utils import *
import re


with open('grids/tetgen_intersections.txt') as f:
    lines = f.readlines()

cells_a = []
cells_b = []
for l in lines:
    match = re.match(r'\s+\((\d+), (\d+), (\d+)\) and \((\d+), (\d+), (\d+)\)\s+', l)
    if match is not None:
        cells_a.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
        cells_b.append((int(match.group(4)), int(match.group(5)), int(match.group(6))))
cells_a = np.array(cells_a)
cells_b = np.array(cells_b)
cells_a = np.sort(cells_a, axis=1)
cells_b = np.sort(cells_b, axis=1)
print('Read', len(cells_a), 'pairs of facets')

points, facets, _ = convertVtkGridToNumpy(readVtkGrid('grids/merged.vtk'))
facets = np.sort(facets, axis=1)








