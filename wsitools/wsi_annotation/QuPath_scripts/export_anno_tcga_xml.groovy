/*
 * Author: Jun Jiang (Jiang.Jun@mayo.edu), tested on QuPath 0.2.0-m2
 *
 * Script to export xml corresponding to all annotations of an Qupath project. Additionaly, a class label text and label ID tabel will be created.
 * The xml meets the TCGA annotation format.  These website contains TCGA xml files
 *   https://datadryad.org/resource/doi:10.5061/dryad.1g2nt41
 *   https://datadryad.org/bitstream/handle/10255/dryad.172216/XML_TCGA_HG.zip?sequence=1
 *       File details: https://datadryad.org/resource/doi:10.5061/dryad.1g2nt41/6
 * Aperio Image Scope(a whole slide image viewer) can also provide annotations in this xml format, which means the xml may also be able to be import to Aperio, haven't test yet.
 *  see also: ./AperioToQuPath.groovy
 * This works by looping through all the images in a project, and checking for the existence of a data file.
 * If a data file is found, read the hierarchy (no need to open the whole image), and write all the annotations to the xml files.
 * If any package is missing, it will throw errors. You need to drag the missing jar onto the QuPath window, and you just need to do this only once. This step copies the jar into QuPath's jar directory
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
// import qupath.lib.roi.PathROIToolsAwt  //old version of QuPath
// import qupath.lib.roi.RoiTools //newer version of QuPath
import qupath.lib.scripting.QP

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.PointsROI
import qupath.lib.roi.PolygonROI
import qupath.lib.roi.AreaROI
import qupath.lib.roi.RectangleROI
import qupath.lib.geom.Point2

import qupath.lib.gui.QuPathGUI
import qupath.lib.io.PathIO
import qupath.lib.objects.classes.PathClass
import qupath.lib.gui.scripting.QPEx


String xmlHead = '''<Annotations MicronsPerPixel="0.252100">
  <Annotation Id="1" Name="" ReadOnly="0" NameReadOnly="0" LineColorReadOnly="0" Incremental="0" Type="4" LineColor="65280" Visible="1" Selected="1" MarkupImagePath="" MacroName="">
    <Attributes>
      <Attribute Name="Description" Id="0" Value=""/>
    </Attributes>
    <Regions>
      <RegionAttributeHeaders>
        <AttributeHeader Id="9999" Name="Region" ColumnWidth="-1"/>
        <AttributeHeader Id="9997" Name="Length" ColumnWidth="-1"/>
        <AttributeHeader Id="9996" Name="Area" ColumnWidth="-1"/>
        <AttributeHeader Id="9998" Name="Text" ColumnWidth="-1"/>
        <AttributeHeader Id="1" Name="Description" ColumnWidth="-1"/>
      </RegionAttributeHeaders>\n'''
String xmlTail = '''    </Regions>
    <Plots/>
  </Annotation>
</Annotations>\n'''

String regionStrHead = '''      <Region Id="%d" Type="%d" Text="%s" GeoShape="%s" Zoom="0.042148" Selected="0" ImageLocation="" ImageFocus="0" Length="74565.8" Area="213363186.2" LengthMicrons="18798.0" AreaMicrons="13560170.4" NegativeROA="0" InputRegionId="0" Analyze="1" DisplayId="1">
        <Attributes/>
        <Vertices>\n'''
String regionStrTail = '''        </Vertices>
      </Region>\n'''

String vertexStr = '''          <Vertex X="%f" Y="%f"/>\n'''


// Get the current project open in QuPath
def project = QPEx.getProject()
if (project == null) {
    print 'No project open!'
    return
}

String root_dir = "H:\\temp"  // modify this root_dir to specify where you would like to save your export

// loop through all the entries to get the annotation label map
File fp = new File(root_dir, "class_label_id.csv")
def Anno_labels = []
for (entry in project.getImageList()) {
    def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
    for (anno in annotations){
        label_txt = anno.getPathClass()  //anno.getPathClass() return the class label
        Anno_labels << label_txt
    }    
}
def labels_set = Anno_labels.toSet()
wrt_str = "Label,ID\n"
labels_set.eachWithIndex{ item, idx -> wrt_str+="${item},${idx}\n"}
fp.write(wrt_str)

// get geometric type (rectangle, polygon, line ...) of the ROI from the class string
def get_geoType(String roi_class_txt){
    def last = roi_class_txt.tokenize('.')[-1]
    return last.replace("ROI","")
}

int img_cnt = 0
// Loop through all the entries in the project
for (entry in project.getImageList()) {
    String img_name = entry.getImageName()
    println(sprintf('Processing %s', img_name))
    
    String uuid = img_name.split("\\.")[0]
    File fh = new File(root_dir,uuid+".xml")
    
    img_cnt += 1
    if (img_cnt > 2){
        break
    }
    
    // Read the hierarchy, but *not* anything else (don't need a full ImageData)
    def hierarchy = entry.readHierarchy()

    int nObjects = hierarchy.nObjects()
    def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}
    // Print the total number of objects and number of annotations
    println(sprintf('- Number of objects: %d, Number of annotations: %d', nObjects, annotations.size()))     
    
    int anno_cnt = 0
    String wrt_str = xmlHead
    for (anno in annotations){
        anno_cnt += 1
        int id = labels_set.findIndexOf {it == anno.getPathClass()}
        
        anno_geo_type = get_geoType(anno.getROI().getClass().toString())
        region_str = String.format(regionStrHead, anno_cnt, id, anno.getPathClass(), anno_geo_type)  //anno.getPathClass() return the class label
        wrt_str += region_str
        // if you would like to export point annotations, uncomment the below lines.
//        if (!(anno.getROI() instanceof PointsROI)){   // annotations are not points for offset calculation
//            points = anno.getROI().getPolygonPoints()
//            for (p in points){
//                str_points = String.format(vertexStr, p.x, p.y)
//                wrt_str += str_points
//            }
//        }
        // if you would like to export polygon annotations, uncomment the four below lines.
//         points = anno.getROI().getPolygonPoints() // getPolygonPoints() has been replaced due to QuPath version upgrade
        points = anno.getROI().getAllPoints()
        for (p in points){
            points_str = String.format(vertexStr, p.x, p.y)
            wrt_str += points_str
        }
        wrt_str += regionStrTail
    }
    wrt_str += xmlTail
    fh.write(wrt_str)
}
print "Done"





