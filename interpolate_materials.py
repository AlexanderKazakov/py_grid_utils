from utils import *
import pickle

# read two grids and transfer cell info (material) from one to another


src = readVtkGrid("grids/initial/mesh_initial.vtk")
dst = readVtkGrid("grids/results/grid_with_materials_from_naively_refined_surfaces_004.vtk")

p = np.zeros((dst.GetNumberOfCells(), 4, 3), dtype=np.float64)
for cnt in range(dst.GetNumberOfCells()):
    for i in range(4):
        p[cnt, i, :] = dst.GetCell(cnt).GetPoints().GetPoint(i)
centroids = np.sum(p, axis=1) / 4
with open('grids/tmp_data.pickle', 'wb') as f:
    pickle.dump(centroids, f)
# with open('grids/tmp_data.pickle', 'rb') as f:
#     centroids = pickle.load(f)

locator = vtk.vtkCellLocator()
locator.SetDataSet(src)
locator.BuildLocator()

cells_info = vtk.vtkIntArray()
cells_info.SetName('cell_info')
cells_info.SetNumberOfComponents(1)
cells_info.Allocate(dst.GetNumberOfCells())
for cnt in range(cells_info.GetSize()):
    cell_in_src = locator.FindCell(centroids[cnt])
    if cell_in_src > 0:
        info = src.GetCellData().GetArray(0).GetValue(cell_in_src)
    else:
        info = 0
    cells_info.InsertNextValue(info)
dst.GetCellData().AddArray(cells_info)
writeVtkGrid(dst, 'grids/results/grid_with_materials.vtk')

points, cells, materials = convertVtkGridToNumpy(dst)
grid_quality(points, cells)
# grid_quality(points, cells[materials[:, 0] != 0])





