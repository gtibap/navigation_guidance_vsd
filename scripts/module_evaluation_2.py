import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import math
import numpy as np
from sklearn.decomposition import PCA
import pickle

#
# module_evaluation_2
#

class module_evaluation_2(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "module_evaluation_2"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#module_evaluation_2">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # module_evaluation_21
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='module_evaluation_2',
    sampleName='module_evaluation_21',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'module_evaluation_21.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='module_evaluation_21.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='module_evaluation_21'
  )

  # module_evaluation_22
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='module_evaluation_2',
    sampleName='module_evaluation_22',
    thumbnailFileName=os.path.join(iconsPath, 'module_evaluation_22.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='module_evaluation_22.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='module_evaluation_22'
  )

#
# module_evaluation_2Widget
#

class module_evaluation_2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/module_evaluation_2.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = module_evaluation_2Logic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      self.ui.applyButton.toolTip = "Compute output volume"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    # try:
    #
    #   # Compute output
    #   self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
    #     self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)
    #
    #   # Compute inverted output (if needed)
    #   if self.ui.invertedOutputSelector.currentNode():
    #     # If additional output volume is selected then result with inverted threshold is written there
    #     self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
    #       self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)
    #
    # except Exception as e:
    #   slicer.util.errorDisplay("Failed to compute results: "+str(e))
    #   import traceback
    #   traceback.print_exc()
    ##################################
    ##################################

    print('Hello: module evaluation 2')
    # list_points = []
    def read_fiducials(node_name):
        list_points_world = []
        node_vsd_fids = slicer.util.getNode(node_name)
        numFids = node_vsd_fids.GetNumberOfFiducials()

        # access to fiducials coordinates
        for i in range(numFids):
          world = [0,0,0,0]
          node_vsd_fids.GetNthFiducialWorldCoordinates(i,world)
          list_points_world.append(world[:-1])
          # ras = [0,0,0]
          # node_fids.GetNthFiducialPosition(i,ras)
          # list_points.append(ras)
        return np.array(list_points_world)

    # pointsVSD1 = read_fiducials('points_H3_VSD1')
    pointsVSD1 = read_fiducials('points_H1_VSD1')
    # print(pointsVSD1)
    # print('size: ', len(pointsVSD1))
    pca = PCA()
    pca.fit(pointsVSD1)
    # normal vector of the plane is the third component of PCA
    # normal vector of the plane is the third component of PCA
    x_=pca.components_[0]
    y_=pca.components_[1]
    z_=pca.components_[2]

    m_=pca.mean_

    # print("axes VSD: x, y, z")
    print('x: ', x_)
    print('y: ', y_)
    print('z: ', z_)
    # print(z_,"; ", 10*z_, "; ", 10*z_+m_)
    print("mean: ", m_)

    # # center and normal vector of VSD heart 1
    vsd_center = np.copy(m_)
    vsd_normal = np.copy(z_)
    x_vsd = np.copy(x_)
    y_vsd = np.copy(y_)

    # intersection plane and line
    # nodePalpationTool = slicer.util.getNode('replayTransform')
    # nodeNeedleProj = slicer.util.getNode("needleProj")
    # node_vsd_center = slicer.util.getNode('vsd_center')
    # node_points_vsd_plane = slicer.util.getNode("points_H1_VSD1_plane")
    # sliceYellowNode = slicer.app.layoutManager().sliceWidget('Yellow').mrmlSliceNode()

    # read landmarks reference and initial heart models
    # ptsReference = read_fiducials('PointsReference')
    # ptsInitial = read_fiducials('PointsInitial')
    # ptsReference = read_fiducials('ReferencePoints')
    # ptsInitial = read_fiducials('InitialPoints')
    ptsReference = read_fiducials('PhysicalHeart')
    ptsInitial = read_fiducials('VirtualHeart')

    print('ptsReference: ')
    print(ptsReference)
    print('ptsInitial: ')
    print(ptsInitial)

    error = np.subtract(ptsReference, ptsInitial)
    mag_error = np.linalg.norm(error, axis=1)
    print('error: ')
    print(error)
    print('mag. error: ')
    print(mag_error)
    # rms error
    rms_error = np.sqrt( np.sum(mag_error*mag_error) / len(mag_error) )
    print('rms_error: ', rms_error)


    # needleTip = [0,0,0,1]
    # needleBase = [0,0,-40,1]
    #
    # node_replayTransform_1 = slicer.util.getNode('replayTransform')
    # node_replayTransform_2 = slicer.util.getNode('replayTransform_2')
    #
    # transformNeedle_1 = vtk.vtkMatrix4x4()
    # transformNeedle_2 = vtk.vtkMatrix4x4()
    #
    # node_replayTransform_1.GetMatrixTransformToWorld(transformNeedle_1)
    # node_replayTransform_2.GetMatrixTransformToWorld(transformNeedle_2)
    # #
    # a_0 = transformNeedle_1.MultiplyPoint(needleTip)
    # a_1 = transformNeedle_1.MultiplyPoint(needleBase)
    #
    # b_0 = transformNeedle_2.MultiplyPoint(needleTip)
    # b_1 = transformNeedle_2.MultiplyPoint(needleBase)
    #
    # a_0 =np.array(a_0[:-1])
    # a_1 =np.array(a_1[:-1])
    # needle_axis_a = a_1 - a_0
    # needle_axis_a = needle_axis_a / np.linalg.norm(needle_axis_a)
    #
    # b_0 =np.array(b_0[:-1])
    # b_1 =np.array(b_1[:-1])
    # needle_axis_b = b_1 - b_0
    # needle_axis_b = needle_axis_b / np.linalg.norm(needle_axis_b)
    #
    # angle_dif=np.arccos(np.dot(needle_axis_a, needle_axis_b))
    # angle_dif_degrees = math.degrees(angle_dif)
    #
    # dif_tip = a_0 - b_0
    # dist= np.linalg.norm(dif_tip)
    #
    # print('dif_tip: ', dif_tip)
    # print('dist, angle_dif(rad), angle_dif(degrees): ', dist, angle_dif, angle_dif_degrees)



    #
    # root_dir='E:/VSD/slicer/selected_data/aug10/'
    # subfolder='test03/'
    # rec=0
    # list_files = np.arange(rec,rec+1)
    #
    # list_points_plane=[]
    # transformNeedle = vtk.vtkMatrix4x4()
    # self.count=0
    # flag=False
    #
    # for self.count in list_files:
    #     filename=str(self.count)
    #     #with open('C:/Users/an40770/Documents/a2021/slicer/record_transformations/interaction/'+ subfolder +'reg'+filename,'rb') as f:
    #     with open(root_dir + subfolder +'reg'+filename,'rb') as f:
    #         c=0
    #         while flag:
    #             # print(filename, c)
    #             needleTip = [0,0,0,1]
    #             needleBase = [0,0,-40,1]
    #             try:
    #                 data = pickle.load(f)
    #                 # print(filename, c)
    #
    #                 transf_matrix=np.array(data[1])
    #                 for i in range(4):
    #                     for j in range(4):
    #                         transformNeedle.SetElement(i,j,transf_matrix[i,j])
    #
    #                 needleTip = transformNeedle.MultiplyPoint(needleTip)
    #                 needleBase = transformNeedle.MultiplyPoint(needleBase)
    #
    #                 needleTip =np.array(needleTip[:-1])
    #                 needleBase=np.array(needleBase[:-1])
    #                 needle_axis = needleBase - needleTip
    #                 needle_axis = needle_axis / np.linalg.norm(needle_axis)
    #                 # print('needle_axis: ', needle_axis)
    #
    #                 # intersection plane and line (inspired by Line-plane intersection, Algebraic form, wikipedia)
    #                 dot_value = np.dot(needle_axis, vsd_normal)
    #                 if abs(dot_value) <= 1e-6:
    #                     d = 1e6
    #                 else:
    #                     d = (np.dot(vsd_center, vsd_normal) - np.dot(needleTip, vsd_normal)) / dot_value
    #
    #                 p_intersection = needleTip + d*needle_axis
    #                 list_points_plane.append(p_intersection)
    #                 # print('{}, {}'.format(dot_value, d))
    #                 # print('p_intersection: ', p_intersection)
    #                 # print('distance: ', d)
    #                 c+=1
    #
    #             except EOFError:
    #                 flag=False
    #
    # # # vsd points boundaries in the plane (without the z_vsd component)
    # # for pt_vsd, id_pt in zip(pointsVSD1, np.arange(len(pointsVSD1))):
    # #     pt_vsd=pt_vsd-vsd_center
    # #     pt_vsd_plane = np.dot(pt_vsd, x_vsd)*x_vsd + np.dot(pt_vsd,y_vsd)*y_vsd + vsd_center
    # #     node_points_vsd_plane.SetNthFiducialPosition(id_pt,  pt_vsd_plane[0], pt_vsd_plane[1], pt_vsd_plane[2])
    #
    # # save to file vsd points boundaries in the plane defined by x_vsd and y_vsd
    # vsd_boundaries = []
    # for pt_vsd in pointsVSD1:
    #     pt_vsd=pt_vsd-vsd_center
    #     vsd_boundaries.append([np.dot(pt_vsd, x_vsd), np.dot(pt_vsd,y_vsd)])
    #
    # # save to file vsd points boundaries in the plane defined by x_vsd and y_vsd
    # pts_plane = []
    # for pt_i in list_points_plane:
    #     pt_i=pt_i-vsd_center
    #     pts_plane.append([np.dot(pt_i, x_vsd), np.dot(pt_i, y_vsd)])
    #
    # # print("vsd boundaries")
    # # print(vsd_boundaries)
    #
    # # with open(root_dir + subfolder +'output_pts.dat','wb') as f:
    # #     pickle.dump([vsd_boundaries, pts_plane], f, pickle.HIGHEST_PROTOCOL)
    #
    # print('Done.')




    # visualize markups for needle tail or base
    # node_vsd_center.SetNthFiducialPosition(0,  vsd_center[0], vsd_center[1], vsd_center[2])
    # nodeNeedleProj.SetNthFiducialPosition(0,  p_intersection[0], p_intersection[1], p_intersection[2])
    # sliceYellowNode.SetSliceToRASByNTP(-vsd_normal[0], -vsd_normal[1], -vsd_normal[2], x_vsd[0], x_vsd[1], x_vsd[2], vsd_center[0], vsd_center[1], vsd_center[2], 0)



#
# module_evaluation_2Logic
#

class module_evaluation_2Logic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

#
# module_evaluation_2Test
#

class module_evaluation_2Test(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_module_evaluation_21()

  def test_module_evaluation_21(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('module_evaluation_21')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = module_evaluation_2Logic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
