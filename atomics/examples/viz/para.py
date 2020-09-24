# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import time
#### disable automatic camera reset on 'Show'
for i in range(315):
    print('i is', i)
    in_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/solutions_iterations/density_'+str(i+1)+'.pvd'
    out_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/solutions_iterations/density'+str(i+1)+'.png'

    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    density_1pvd = PVDReader(FileName=in_path)
    density_1pvd.CellArrays = ['density']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [1022, 799]

    # get layout
    layout1 = GetLayout()

    # show data in view
    density_1pvdDisplay = Show(density_1pvd, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'density'
    densityLUT = GetColorTransferFunction('density')
    densityLUT.RGBPoints
    densityLUT.RGBPoints = [0.21352491812111066, 0.231373, 0.298039, 0.752941, 0.429009605694789, 0.865003, 0.865003, 0.865003, 0.6444942932684674, 0.705882, 0.0156863, 0.14902]
    densityLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'density'
    densityPWF = GetOpacityTransferFunction('density')
    densityPWF.Points = [0.21352491812111066, 0.0, 0.5, 0.0, 0.6444942932684674, 1.0, 0.5, 0.0]
    densityPWF.ScalarRangeInitialized = 1


    # rescale color and/or opacity maps used to exactly fit the current data range
    density_1pvdDisplay.RescaleTransferFunctionToDataRange(False, True)

    # rescale color and/or opacity maps used to exactly fit the current data range
    density_1pvdDisplay.RescaleTransferFunctionToDataRange(False, True)









    # trace defaults for the display properties.
    density_1pvdDisplay.Representation = 'Surface'
    density_1pvdDisplay.ColorArrayName = ['CELLS', 'density']
    density_1pvdDisplay.LookupTable = densityLUT
    density_1pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    density_1pvdDisplay.SelectOrientationVectors = 'None'
    density_1pvdDisplay.ScaleFactor = 16.0
    density_1pvdDisplay.SelectScaleArray = 'density'
    density_1pvdDisplay.GlyphType = 'Arrow'
    density_1pvdDisplay.GlyphTableIndexArray = 'density'
    density_1pvdDisplay.GaussianRadius = 0.8
    density_1pvdDisplay.SetScaleArray = [None, '']
    density_1pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    density_1pvdDisplay.OpacityArray = [None, '']
    density_1pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    density_1pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    density_1pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    density_1pvdDisplay.ScalarOpacityFunction = densityPWF
    density_1pvdDisplay.ScalarOpacityUnitDistance = 12.139244620058346

    # reset view to fit data
    renderView1.ResetCamera()

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    density_1pvdDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [80.0, 40.0, 345.58012940880167]
    renderView1.CameraFocalPoint = [80.0, 40.0, 0.0]
    renderView1.CameraParallelScale = 89.44271909999159

    # save screenshot
    SaveScreenshot('/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/solutions_iterations/density1.png', renderView1, ImageResolution=[1022, 799],
        TransparentBackground=1, 
        # PNG options
        CompressionLevel='3')

    #### saving camera placements for all active views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [80.0, 40.0, 345.58012940880167]
    renderView1.CameraFocalPoint = [80.0, 40.0, 0.0]
    renderView1.CameraParallelScale = 89.44271909999159

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).

    # save screenshot
    SaveScreenshot(out_path, renderView1, ImageResolution=[844, 540],
        TransparentBackground=1, 
        # PNG options
        CompressionLevel='3')