// directly extract patches from QuPath

import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.GeometryROI
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.PathObject
import static qupath.lib.gui.scripting.QPEx.*

int patchSize = 512  // Adjust this as needed
int downsample = 5  // downsample rate
def outputDir = buildFilePath(PROJECT_BASE_DIR, 'patches')
mkdirs(outputDir)

def imageData = getCurrentImageData()
def server = getCurrentImageData().getServer()
def hierarchy = imageData.getHierarchy()
int width = server.getWidth()
int height = server.getHeight()

//def server = QP.getCurrentServer()
def cal = server.getPixelCalibration()
String xUnit = cal.getPixelWidthUnit()
String yUnit = cal.getPixelHeightUnit()
double pixelWidth = cal.getPixelWidthMicrons()
double pixelHeight = cal.getPixelHeightMicrons()

def firstAnnotation = getAnnotationObjects().findAll{it.getROI().getRoiName() == 'Rectangle'}
removeObjects(firstAnnotation, true)

for (int y = 0; y < height; y += patchSize) {
    for (int x = 0; x < width; x += patchSize) {
        
        // Create a rectangle ROI for the current patch
        def patchROI = new RectangleROI(x, y, patchSize, patchSize)
        def annotation = new PathAnnotationObject(patchROI, PathClass.fromString("Patch"))

        imageData.getHierarchy().addObject(annotation, false)
        
        print("added")
    }
}
fireHierarchyUpdate()

def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
def Polygon_annotation_list = []
for (anno in annotations){
    if(anno.getROI() instanceof GeometryROI){
        Polygon_annotation_list << anno
    }
    else if(anno.getROI() instanceof PolygonROI){
        Polygon_annotation_list << anno
    }
}

int annotated_cnt = 1
hierarchy.getAnnotationObjects().eachWithIndex{ anno, idx ->
        roi = anno.getROI()
        
        if (anno.getPathClass().toString().equals("Patch")) {
            x = roi.centroidX
            y = roi.centroidY
            
            for (polygon in Polygon_annotation_list){
                if (polygon.getROI().contains(x, y)){
                    println(sprintf('\t\t -Patch is labeled by a polygon: %f,%f(%s), Label: %s', x*pixelWidth, y*pixelHeight, xUnit, polygon.getPathClass().toString()))
                    anno.setPathClass(polygon.getPathClass())
                    annotated_cnt += 1
                    
                    def requestROI = RegionRequest.createInstance(server.getPath(), downsample, roi)
                    filename =  String.format( "%s_%s_%s.tif",anno.getPathClass().toString(),x,y)
                    print(filename)
                    writeImageRegion(server, requestROI, buildFilePath(PROJECT_BASE_DIR, 'patches', filename))
                }
            }

        }
}  

def secondAnnotation = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("Patch")}
removeObjects(secondAnnotation, true)
        
print(sprintf("\t\t -Annotated patch count in total: %d", annotated_cnt))       
        
        
        
        
        
