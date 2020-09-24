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
renderView1.ViewSize = [4220, 2700]

# get layout
layout1 = GetLayout()

# show data in view
stiffness_genpvdDisplay = Show(stiffness_genpvd, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'density'
densityLUT = GetColorTransferFunction('density')
densityLUT.RGBPoints = [0.14999999999999986, 0.231373, 0.298039, 0.752941, 0.5750000000000002, 0.865003, 0.865003, 0.865003, 1.0000000000000004, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'density'
densityPWF = GetOpacityTransferFunction('density')
densityPWF.Points = [0.14999999999999986, 0.0, 0.5, 0.0, 1.0000000000000004, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
stiffness_genpvdDisplay.Representation = 'Surface'
stiffness_genpvdDisplay.ColorArrayName = ['CELLS', 'density']
stiffness_genpvdDisplay.LookupTable = densityLUT
stiffness_genpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.SelectOrientationVectors = 'None'
stiffness_genpvdDisplay.ScaleFactor = 0.24
stiffness_genpvdDisplay.SelectScaleArray = 'density'
stiffness_genpvdDisplay.GlyphType = 'Arrow'
stiffness_genpvdDisplay.GlyphTableIndexArray = 'density'
stiffness_genpvdDisplay.GaussianRadius = 0.012
stiffness_genpvdDisplay.SetScaleArray = [None, '']
stiffness_genpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.OpacityArray = [None, '']
stiffness_genpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
stiffness_genpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
stiffness_genpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
stiffness_genpvdDisplay.ScalarOpacityFunction = densityPWF
stiffness_genpvdDisplay.ScalarOpacityUnitDistance = 0.09447630744140174

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
stiffness_genpvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get color legend/bar for densityLUT in view renderView1
densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
densityLUTColorBar.Orientation = 'Horizontal'
densityLUTColorBar.WindowLocation = 'AnyLocation'
densityLUTColorBar.Position = [0.2707109004739336, 0.05777777777777774]
densityLUTColorBar.Title = 'density'
densityLUTColorBar.ComponentTitle = ''
densityLUTColorBar.ScalarBarLength = 0.3300000000000008

# change scalar bar placement
densityLUTColorBar.Position = [0.2979620853080568, 0.1855555555555555]
densityLUTColorBar.ScalarBarLength = 0.3300000000000015

# Properties modified on densityLUTColorBar
densityLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
densityLUTColorBar.LabelColor = [0.0, 0.0, 0.0]

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [1.2, 0.4, 4.887241058965766]
renderView1.CameraFocalPoint = [1.2, 0.4, 0.0]
renderView1.CameraParallelScale = 1.2649110640673518

# save screenshot
SaveScreenshot('/home/jyan_linux/Desktop/1.png', renderView1, ImageResolution=[844, 540],
    TransparentBackground=1, 
    # PNG options
    CompressionLevel='3')