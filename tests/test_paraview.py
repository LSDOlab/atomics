# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
stiffness_genpvd = PVDReader(FileName='/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/solutions/stiffness_gen.pvd')
stiffness_genpvd.CellArrays = ['density']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [1266, 810]

# get layout
layout1 = GetLayout()

# show data in view
stiffness_genpvdDisplay = Show(stiffness_genpvd, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')

# trace defaults for the display properties.
stiffness_genpvdDisplay.Representation = 'Surface'
stiffness_genpvdDisplay.ColorArrayName = ['CELLS', 'density']
stiffness_genpvdDisplay.LookupTable = densityLUT
stiffness_genpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.SelectOrientationVectors = 'None'
stiffness_genpvdDisplay.ScaleFactor = 16.0
stiffness_genpvdDisplay.SelectScaleArray = 'density'
stiffness_genpvdDisplay.GlyphType = 'Arrow'
stiffness_genpvdDisplay.GlyphTableIndexArray = 'density'
stiffness_genpvdDisplay.GaussianRadius = 0.8
stiffness_genpvdDisplay.SetScaleArray = [None, '']
stiffness_genpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.OpacityArray = [None, '']
stiffness_genpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
stiffness_genpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
stiffness_genpvdDisplay.ScalarOpacityFunction = densityPWF
stiffness_genpvdDisplay.ScalarOpacityUnitDistance = 12.086595437738142

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [80.0, 40.0, 10000.0]
renderView1.CameraFocalPoint = [80.0, 40.0, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
stiffness_genpvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)

# change scalar bar placement
densityLUTColorBar.Orientation = 'Horizontal'
densityLUTColorBar.WindowLocation = 'AnyLocation'
densityLUTColorBar.Position = [0.21383886255924128, 0.10592592592592631]
densityLUTColorBar.ScalarBarLength = 0.32999999999999946

# change scalar bar placement
densityLUTColorBar.ScalarBarLength = 0.4935071090047389

# change scalar bar placement
densityLUTColorBar.Position = [0.17118483412322238, 0.045]
densityLUTColorBar.ScalarBarLength = 0.4935071090047388

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [80.0, 40.0, 10000.0]
renderView1.CameraFocalPoint = [80.0, 40.0, 0.0]
renderView1.CameraParallelScale = 73.91960256197652

# save screenshot
SaveScreenshot('python_paraview.png', renderView1, ImageResolution=[844, 540],
    TransparentBackground=1)