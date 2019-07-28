import vtk

# Convert arbitrary vtk unsrtuctured grid to (surface only!) stl format


# filename = 'grids/tmp_facets'
# filename = 'grids/tmp_cells'
filename = "grids/surfaces"

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(filename + '.vtk')

surface_filter = vtk.vtkDataSetSurfaceFilter()
surface_filter.SetInputConnection(reader.GetOutputPort())

triangle_filter = vtk.vtkTriangleFilter()
triangle_filter.SetInputConnection(surface_filter.GetOutputPort())

writer = vtk.vtkSTLWriter()
writer.SetFileName(filename + '.stl')
writer.SetInputConnection(triangle_filter.GetOutputPort())
writer.Write()







