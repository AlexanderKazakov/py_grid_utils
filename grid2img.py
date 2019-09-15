from utils import *


src_grid = readVtkGrid('grids/results/big_conn_comps.vtk')
points, cells, materials = convertVtkGridToNumpy(src_grid)
materials = materials[0]

r1 = np.min(points, axis=0)
r2 = np.max(points, axis=0)
shift = (r2 - r1) * 0.01
r1 -= shift
r2 += shift
size_x = 500
size_y = int(np.ceil(size_x * (r2[1] - r1[1]) / (r2[0] - r1[0])))
size_z = int(np.ceil(size_x * (r2[2] - r1[2]) / (r2[0] - r1[0])))
img = np.zeros((size_x, size_y, size_z), np.uint8)
step = (r2 - r1) / np.array([size_x, size_y, size_z])
print('img.shape: {} \nstep: {} '.format(img.shape, step))

locator = vtk.vtkCellLocator()
locator.SetDataSet(src_grid)
locator.BuildLocator()
for i in range(size_x):
    for j in range(size_y):
        for k in range(size_z):
            point = r1 + step * np.array([i, j, k]) + step / 2
            cell_in_src = locator.FindCell(point)
            if cell_in_src >= 0:
                img[i, j, k] = materials[cell_in_src]


# img = np.zeros((100, 100, 100), np.uint8)
# img[:50] = 1
# img[:, :50] += 2

write_vti(img, 'grids/img.vti')












