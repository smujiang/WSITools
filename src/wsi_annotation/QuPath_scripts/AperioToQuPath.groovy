/*
* Copied from https://gist.github.com/DanaCase/9cfc23912fee48e437af03f97763d78e
* Convert the annotation file (xml files) from Aperio Image Scope, so that the annotation can be loaded into QuPath
* Also check this link: https://groups.google.com/forum/#!searchin/qupath-users/import$20annotation%7Csort:date/qupath-users/xhCx_nhbWQQ/0kW38lEXCAAJ
*/


import qupath.lib.scripting.QP
import qupath.lib.geom.Point2
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.ImageServer


//Aperio Image Scope displays images in a different orientation
def rotated = true

def server = QP.getCurrentImageData().getServer()
def h = server.getHeight()
def w = server.getWidth()

// need to add annotations to hierarchy so qupath sees them
def hierarchy = QP.getCurrentHierarchy()


//Prompt user for exported aperio image scope annotation file
def file = getQuPath().getDialogHelper().promptForFile('xml', null, 'aperio xml file', null)
def text = file.getText()

def list = new XmlSlurper().parseText(text)


list.Annotation.each {

    it.Regions.Region.each { region ->

        def tmp_points_list = []

        region.Vertices.Vertex.each{ vertex ->

            if (rotated) {
                X = vertex.@Y.toDouble()
                Y = h - vertex.@X.toDouble()
            }
            else {
                X = vertex.@X.toDouble()
                Y = vertex.@Y.toDouble()
            }
            tmp_points_list.add(new Point2(X, Y))
        }

        def roi = new PolygonROI(tmp_points_list)

        def annotation = new PathAnnotationObject(roi)

        hierarchy.addPathObject(annotation, false)
    }
}