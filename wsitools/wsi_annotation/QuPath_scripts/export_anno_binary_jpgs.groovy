/**
 * Script to export binary masks corresponding to all annotations of an image,
 * optionally along with extracted image regions.
 *
 * Note: Pay attention to the 'downsample' value to control the export resolution!
 *
 * @author Pete Bankhead
 * modified by Jun Jiang (Jiang.Jun@mayo.edu), tested on QuPath 0.2.0-m2
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QP

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = QP.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 128.0
// modify this root_dir to specify where you would like to save your export
def pathOutput = "H:\\PenMarking\\eval\\qupath_annotation\\tiles"

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as JPG/PNG
def imageExportType = 'JPG'

// Export each annotation
annotations.each {
    saveImageAndMask(pathOutput, server, it, downsample, imageExportType)
}
print 'Done!'

/**
 * Save extracted image region & mask corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImageAndMask(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    // Extract ROI & classification name
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }

    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s_%s_(%.2f,%d,%d,%d,%d)',
            server.getShortServerName(),
            classificationName,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = PathROIToolsAwt.getShape(roi)
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())
    g2d.fill(shape)
    g2d.dispose()

    // Create filename & export
    if (imageExportType != null) {
        pathOutput = pathOutput + File.separator + server.getShortServerName().take(server.getShortServerName().lastIndexOf('.'))
        File newDir = new File(pathOutput)
        if (!newDir.exists()) {
            newDir.mkdirs()
        }
        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
        ImageIO.write(img, imageExportType, fileImage)
    }
    // Export the mask
    def fileMask = new File(pathOutput, name + '-mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)
}