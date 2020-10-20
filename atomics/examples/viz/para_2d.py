# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
for i in range(742):
    print('i is', i)
    in_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/atomics/examples/solutions_iterations/density_'+str(i+1)+'.pvd'
    out_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/atomics/examples/solutions_iterations/density_'+str(i+1)+'.png'
    # trace generated using paraview version 5.8.1
    #
    # To ensure correct image size when batch processing, please search 
    # for and uncomment the line `# renderView*.ViewSize = [*,*]`

    #### import the simple module from the paraview
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    density_259pvd = PVDReader(FileName=in_path)
    density_259pvd.CellArrays = ['density']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    renderView1.ViewSize = [3006, 3337]

    # get layout
    layout1 = GetLayout()

    # show data in view
    density_259pvdDisplay = Show(density_259pvd, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'density'
    densityLUT = GetColorTransferFunction('density')
    densityLUT.RGBPoints = [0.03499999999999997, 0.231373, 0.298039, 0.752941, 0.5265150755219273, 0.865003, 0.865003, 0.865003, 1.0180301510438547, 0.705882, 0.0156863, 0.14902]
    densityLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'density'
    densityPWF = GetOpacityTransferFunction('density')
    densityPWF.Points = [0.03499999999999997, 0.0, 0.5, 0.0, 1.0180301510438547, 1.0, 0.5, 0.0]
    densityPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    density_259pvdDisplay.Representation = 'Surface'
    density_259pvdDisplay.ColorArrayName = ['CELLS', 'density']
    density_259pvdDisplay.LookupTable = densityLUT
    density_259pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    density_259pvdDisplay.SelectOrientationVectors = 'None'
    density_259pvdDisplay.ScaleFactor = 0.48
    density_259pvdDisplay.SelectScaleArray = 'density'
    density_259pvdDisplay.GlyphType = 'Arrow'
    density_259pvdDisplay.GlyphTableIndexArray = 'density'
    density_259pvdDisplay.GaussianRadius = 0.024
    density_259pvdDisplay.SetScaleArray = [None, '']
    density_259pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    density_259pvdDisplay.OpacityArray = [None, '']
    density_259pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    density_259pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    density_259pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    density_259pvdDisplay.ScalarOpacityFunction = densityPWF
    density_259pvdDisplay.ScalarOpacityUnitDistance = 0.1889526148828035

    # reset view to fit data
    renderView1.ResetCamera()

    #changing interaction mode based on data extents
    renderView1.CameraPosition = [2.4, 0.8, 10000.0]
    renderView1.CameraFocalPoint = [2.4, 0.8, 0.0]

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # show color bar/color legend
    density_259pvdDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # rescale color and/or opacity maps used to exactly fit the current data range
    density_259pvdDisplay.RescaleTransferFunctionToDataRange(False, True)

    # get color legend/bar for densityLUT in view renderView1
    densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
    densityLUTColorBar.Orientation = 'Horizontal'
    densityLUTColorBar.WindowLocation = 'AnyLocation'
    densityLUTColorBar.Position = [0.376140350877193, 0.14741573033707872]
    densityLUTColorBar.Title = 'density'
    densityLUTColorBar.ComponentTitle = ''
    densityLUTColorBar.ScalarBarLength = 0.33000000000000007

    # change scalar bar placement
    densityLUTColorBar.ScalarBarLength = 0.4880041580041581

    # change scalar bar placement
    densityLUTColorBar.Position = [0.270111244848087, 0.21483146067415737]

    # change scalar bar placement
    densityLUTColorBar.ScalarBarLength = 0.616902286902287

    # change scalar bar placement
    densityLUTColorBar.Position = [0.191109165846008, 0.16670411985018734]

    # Properties modified on density_259pvdDisplay
    density_259pvdDisplay.EdgeColor = [1.0, 1.0, 1.0]

    # Properties modified on renderView1
    renderView1.Background = [1.0, 1.0, 1.0]

    densityLUTColorBar = GetScalarBar(densityLUT, renderView1)
    densityLUTColorBar.Title = 'density'
    densityLUTColorBar.ComponentTitle = ''

    # Properties modified on densityLUTColorBar
    densityLUTColorBar.TitleColor = [0.0, 0.0, 0.0]
    densityLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
    # Properties modified on renderView1
    renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]

    # Properties modified on renderView1
    renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]

    #### saving camera placements for all active views

    # current camera placement for renderView1
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [2.4, 0.8, 10000.0]
    renderView1.CameraFocalPoint = [2.4, 0.8, 0.0]
    renderView1.CameraParallelScale = 2.8085759177212717

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).

    # save screenshot
    SaveScreenshot(out_path, renderView1, ImageResolution=[570, 534], 
        # PNG options
        CompressionLevel='0')

    Delete(density_259pvd)
    del density_259pvd

    #### uncomment the following to render all views
    # RenderAllViews()
    # alternatively, if you want to write images, you can use SaveScreenshot(...).