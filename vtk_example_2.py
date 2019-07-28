import vtk


colors = vtk.vtkNamedColors()

filename = 'grids/tmp.stl'

# source = vtk.vtkSphereSource()
# source.SetPhiResolution(16)
# source.SetThetaResolution(16)
source = vtk.vtkCubeSource()
source.Update()

# Write the stl file to disk
stlWriter = vtk.vtkSTLWriter()
stlWriter.SetFileName(filename)
stlWriter.SetInputConnection(source.GetOutputPort())
stlWriter.Write()

# Read and display for verification
reader = vtk.vtkSTLReader()
reader.SetFileName(filename)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.SetSize(1000, 1000)
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)
ren.SetBackground(colors.GetColor3d('cobalt_green'))

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()










