# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
for i in range(1075):
    print('i is', i)
    in_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/solutions_iterations_3d/density_'+str(i+1)+'.pvd'
    out_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/solutions_iterations_3d/density_'+str(i+1)+'.png'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    density_1075pvd = PVDReader(FileName=in_path)
    density_1075pvd.CellArrays = ['density']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [984, 801]
    densityLUT = GetColorTransferFunction('density')
    densityPWF = GetOpacityTransferFunction('density')



    # get layout
    layout1 = GetLayout()

    # show data in view
    density_1075pvdDisplay = Show(density_1075pvd, renderView1, 'UnstructuredGridRepresentation')

    densityLUT.RescaleTransferFunction(9.999999999999988e-05, 1.0000000000000022)

    # Rescale transfer function
    densityPWF.RescaleTransferFunction(9.999999999999988e-05, 1.0000000000000022)

    # rescale color and/or opacity maps used to exactly fit the current data range
    density_1075pvdDisplay.RescaleTransferFunctionToDataRange(True, True)

    # get color transfer function/color map for 'density'
    densityLUT = GetColorTransferFunction('density')
    densityLUT.RGBPoints = [9.999999999999986e-05, 0.231373, 0.298039, 0.752941, 0.5000500000000011, 0.865003, 0.865003, 0.865003, 1.0000000000000022, 0.705882, 0.0156863, 0.14902]
    densityLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'density'
    densityPWF = GetOpacityTransferFunction('density')
    densityPWF.Points = [9.999999999999986e-05, 0.0, 0.5, 0.0, 1.0000000000000022, 1.0, 0.5, 0.0]
    densityPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    density_1075pvdDisplay.Representation = 'Surface'
    density_1075pvdDisplay.ColorArrayName = ['CELLS', 'density']
    density_1075pvdDisplay.LookupTable = densityLUT
    density_1075pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    density_1075pvdDisplay.SelectOrientationVectors = 'None'
    density_1075pvdDisplay.ScaleFactor = 16.0
    density_1075pvdDisplay.SelectScaleArray = 'density'
    density_1075pvdDisplay.GlyphType = 'Arrow'
    density_1075pvdDisplay.GlyphTableIndexArray = 'density'
    density_1075pvdDisplay.GaussianRadius = 0.8
    density_1075pvdDisplay.SetScaleArray = [None, '']
    density_1075pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    density_1075pvdDisplay.OpacityArray = [None, '']
    density_1075pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    density_1075pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    density_1075pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    density_1075pvdDisplay.ScalarOpacityFunction = densityPWF
    density_1075pvdDisplay.ScalarOpacityUnitDistance = 7.11015696922591

    # reset view to fit data
    renderView1.ResetCamera()

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    density_1075pvdDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # Properties modified on densityLUT
    densityLUT.EnableOpacityMapping = 1

    # Properties modified on renderView1
    renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]

    # Properties modified on renderView1
    renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]

    # Properties modified on renderView1
    renderView1.Background = [1.0, 1.0, 1.0]

    # get color legend/bar for densityLUT in view renderView1
    densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
    densityLUTColorBar.Title = 'density'
    densityLUTColorBar.ComponentTitle = ''

    # Properties modified on densityLUTColorBar
    densityLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    densityLUTColorBar.LabelColor = [0.0, 0.0, 0.0]

    # change scalar bar placement
    densityLUTColorBar.Orientation = 'Horizontal'
    densityLUTColorBar.WindowLocation = 'AnyLocation'
    # densityLUTColorBar.Position = [0.244561403508772, 0.00621722846441953]
    densityLUTColorBar.ScalarBarLength = 0.3299999999999999

    # change scalar bar placement
    densityLUTColorBar.ScalarBarLength = 0.43

    # change scalar bar placement
    densityLUTColorBar.Position = [0.2535087719298246, 0.0774531835205993]

    # current camera placement for renderView1
    renderView1.CameraPosition = [-214.42178613249212, 60.91969320414165, 185.7678323424357]
    renderView1.CameraFocalPoint = [80.00000000000014, 40.00000000000019, 4.999999999999855]
    renderView1.CameraViewUp = [0.04191753739854372, 0.9980044340260331, -0.04722361403809724]
    renderView1.CameraParallelScale = 89.58236433584459

    # save screenshot
    SaveScreenshot(out_path, renderView1, ImageResolution=[570, 534], 
        # PNG options
        CompressionLevel='0')

    #### saving camera placements for all active views

    # current camera placement for renderView1
    renderView1.CameraPosition = [-214.42178613249212, 60.91969320414165, 185.7678323424357]
    renderView1.CameraFocalPoint = [80.00000000000014, 40.00000000000019, 4.999999999999855]
    renderView1.CameraViewUp = [0.04191753739854372, 0.9980044340260331, -0.04722361403809724]
    renderView1.CameraParallelScale = 89.58236433584459

    Delete(density_1075pvd)
    del density_1075pvd

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).