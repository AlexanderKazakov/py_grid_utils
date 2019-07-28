#### For mesh conversion, see 
https://github.com/nschloe/meshio

#### tetgen calls examples:
```
C:\DATA\Programs\tetgen\build_dir\Release\tetgen.exe -p -q1.2/15 -a10.0 -Y -k -V skull.stl
```
where from my experience:
- -p to mesh surface from stl file 
- -q{1}/{2} {1} from 1.2 to 2, {2} from 14 to 18
- -a{1} {1} -- linear tetra size
- -Y "to preserve surface"
- -k output to vtk
- -V verbose

file *.stl for tetgen must be a closed surface

 #### CGAL useful utils:
 - https://doc.cgal.org/latest/Polygon_mesh_processing/index.html#HFExample_2  
 -- fill all the holes in a surface to make it closed surface
 - https://doc.cgal.org/latest/Polygon_mesh_processing/index.html#RemeshingExample_1
 -- remesh the surface -- make more homogeneous triangles, improve quality
 
For remeshing, surface *.off file for CGAL must be oriented, see ```orient_facets``` function

#### My scripts:
- utils.py -- some useful functions to work with grids in python
- interpolate_materials.py -- restore cells info to a new grid from the initial grid using KD-tree from vtk  
- remove_crossing_facets.py -- remove facets that are outer for some non-convex hull from a surface
- my_vtk_to_vlad_mesh.py -- convert vtk to DGM by Vlad format
- merge_surfaces.py
- refine_surfaces.py, extract_connected_components.py -- struggling to improve quality of skull mesh
- grid2surfaces.py -- extract surfaces that split one material from another, 
i.e for 3 materials there will be 6 surfaces (count outer space as material -1) 