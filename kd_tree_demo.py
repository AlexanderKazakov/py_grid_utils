from utils import *


# kd tree
points = vtk.vtkPoints()
points.InsertNextPoint(0, 0, 0)
points.InsertNextPoint(1, 0, 0)
points.InsertNextPoint(1, 1, 0)
points.InsertNextPoint(0, 1, 0)

kd = vtk.vtkKdTree()
kd.BuildLocatorFromPoints(points)

dist = 0.0
dist_ref = vtk.reference(dist)
print('point id =', kd.FindClosestPoint(0.51, 0, 0, dist_ref))
print('distance =', np.sqrt(dist_ref.get()))


# for unsructured grid:
grid = readVtkGrid("grids/surfaces.1.vtk")
kd.BuildLocatorFromPoints(grid)
print(kd.FindClosestPoint(grid.GetPoint(281542), dist_ref))
print(kd.GetNumberOfRegions())


# cell locator
src = vtk.vtkUnstructuredGrid()
src.Allocate(1)
src.InsertNextCell(vtk.VTK_TETRA, 4, [0, 1, 2, 3])
vtk_points = vtk.vtkPoints()
vtk_points.InsertPoint(0, [0, 0, 0])
vtk_points.InsertPoint(1, [1, 0, 0])
vtk_points.InsertPoint(2, [0, 1, 0])
vtk_points.InsertPoint(3, [0, 0, 1])
src.SetPoints(vtk_points)

locator = vtk.vtkCellLocator()
locator.SetDataSet(src)
locator.BuildLocator()
print(locator.FindCell([0.1, 0.1, 0.1]))
ids = vtk.vtkIdList()
locator.FindCellsAlongLine([0.1, 0.1, 0.1], [0.1, 0.1, 0.1000001], 0.001, ids)
print(ids.GetId(0))






