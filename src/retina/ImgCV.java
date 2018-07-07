/**
 * @author Ondřej Končal
 * @date 29.11.2016
 */
package retina;

import java.awt.FlowLayout;
import java.awt.Rectangle;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import static org.opencv.core.Core.split;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Class which implements finding hemorrhages in retina image <br>
 * Requires instaled OpenCV
 * 
 * @author Ondřej Končal
 */
public class ImgCV {
    /**
     * Original loaded image
     */
    private BufferedImage image;
    /**
     * Original resized image to (1024 x 768) and converted to Mat
     */
    private Mat imageMat;
    /**
     * Masking matrix with value 0 for real retina image (without border)
     */
    private Mat maskMat;
    /**
     * Masking matrix with value 0 for Optic disk
     */
    private Mat discMat;
    /**
     * Value for treshold which finds border of retina <br>
     * !!! Works only for near black border !!!
     */
    private static final int MASK_THRESHOLD = 10;
    /**
     * Maximal value of pixel
     */
    private static final int MAX_PIXEL_VALUE = 255;
    /**
     * minimal value of pixel
     */
    private static final int MIN_PIXEL_VALUE = 0;
    
    /**
     * Load image from given path and convert it to Mat resized to (1024 x 768). If load fail, program is terminated and return 0.
     * 
     * @param path full path of image to work with
     */
    public ImgCV(String path){
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
        
        image=null;
        maskMat=null;
        discMat=null;
       
        File file = new File(path);
        try{
            image = ImageIO.read(file);
        }catch(IOException e){}
        
        if(image==null){
            System.out.println("Failed to load image");
            System.exit(0);
        }
        
        if(image!=null){
            imageMat = bufferedImageToMat(image);
            //resize
            Imgproc.resize(imageMat, imageMat, new Size(1024,768));
        }
    }
    
    //Imgproc.cvtColor(imageMat,imageGrayMat,Imgproc.COLOR_RGB2GRAY);
    //Imgproc.equalizeHist(imageGrayMat, equHist);
    //Imgproc.GaussianBlur(equHist, gaussBlur, new Size(3,3), 0);
    //Imgproc.Canny(equHist, imageCannyMat, 100, 300);
    
    //**************************************************
    //*******Hemorrhages & CWS detection****************
    //**************************************************
    /**
     * Compute and show image with found and marked hemorrhages places (blue) and CWS (green)
     */
    public void findDiseases(){
        displayMatrix(imageMat, "Original Image");
        
        Mat preprocMat = preprocessing();
        //displayMatrix(preprocMat, "Preprocessed Image");
        Mat claheMat = CLAHE(preprocMat);
        //displayMatrix(claheMat, "CLAHE Image");
        
        Mat sureMaskMat = opticDiscRecognition(claheMat);
        //displayMatrix(discMat, "Disc Image");
        //displayMatrix(sureMaskMat, "Big spot Image");
        
        Mat nobgMat = removeBackground(inversion(claheMat), 15);
        //displayMatrix(nobgMat, "No bg Image");
        
        //add OD to image
        ArrayList<Point> lp = new ArrayList<>();
        for(int i=0; i<nobgMat.rows(); i++){
            for(int j=0; j<nobgMat.cols(); j++){
                if(discMat.get(i, j)[0]==MAX_PIXEL_VALUE){
                    nobgMat.put(i, j, MIN_PIXEL_VALUE);
                    lp.add(new Point(i,j));
                }
            }
        }
        Point midPoint;
        if(!lp.isEmpty()){
            midPoint = lp.get(lp.size()/2);   //point in square
        }
        else{
            midPoint = new Point(0,0);
        }
        //displayMatrix(nobgMat, "With disc Image");
        
        Mat mainNoVein = floodFromOD(nobgMat, midPoint);
        //displayMatrix(mainNoVein, "Main vein removed Image");
        
        Mat restVein = veinRemove(mainNoVein);
        //displayMatrix(restVein, "Line Image");
        
        //remove 'all' veins
        Mat allDisMat = mainNoVein.clone();
        for(int i=0; i<mainNoVein.rows(); i++){
            for(int j=0; j<mainNoVein.cols(); j++){
                if(restVein.get(i,j)[0]==MAX_PIXEL_VALUE)
                    allDisMat.put(i,j,MAX_PIXEL_VALUE);
                else
                    allDisMat.put(i,j,mainNoVein.get(i, j)[0]);
            }
        }
        Imgproc.medianBlur(allDisMat, allDisMat, 3);
        //displayMatrix(allDisMat, "Without vein Image");
        
        //----------------------------------------------------
        //CWS
        Mat potentialMat = potentialCWS(claheMat);
        //displayMatrix(potentialMat, "Potential CWS Image"); 
        
        Mat minMat = minArea(potentialMat);
        //displayMatrix(minMat, "Minimal Image"); 
            
        Mat cwsMat = CWS(minMat, sureMaskMat);
        //displayMatrix(cwsMat, "Cotton Wool Spots Image");
        
        ArrayList cwsList = getCWSPoints(cwsMat);
        
        //-----------------------------------------------------
        Mat noCWSMat = removeCWS(allDisMat, cwsList);
        //displayMatrix(noCWSMat, "noCWS Image");
        
        //add it to CWS mat
        Mat finalMat = twoDiseaseRecognition(cwsMat, noCWSMat);
        displayMatrix(finalMat, "FINAL Image");
    }
    
    //**************************************************
    //*************CWS detection************************
    //**************************************************
    /**
     * Compute and show image with found and marked CWS places
     */
    public void findCWS(){
        Mat preprocMat = preprocessing();
        //displayMatrix(preprocMat, "Preprocessed Image");
        
        Mat claheMat = CLAHE(preprocMat);
        //displayMatrix(claheMat, "CLAHE Image");
        
        Mat sureMaskMat = opticDiscRecognition(claheMat);
        //displayMatrix(discMat, "Disc Image");
        //displayMatrix(sureMaskMat, "Big spot Image");
        
        Mat potentialMat = potentialCWS(claheMat);
        //displayMatrix(potentialMat, "Potential CWS Image"); 
        
        Mat minMat = minArea(potentialMat);
        //displayMatrix(minMat, "Minimal Image"); 
            
        Mat cwsMat = CWS(minMat, sureMaskMat);
        displayMatrix(cwsMat, "Cotton Wool Spots Image");
        
        //getCWSPoints(cwsMat);
    }
    
    /**
     * To matrix m is added blue pixel for each one in mask and removed all disturbing pixels in border of retina<br>
     * Return null on 'm' == null or 'mask' == null
     * 
     * @param m 3-channel matrix
     * @param mask mask for coloring
     * @return Matrix with blue pixels on mask
     */
    private Mat twoDiseaseRecognition(Mat m, Mat mask){
        if(m==null || mask==null)
            return null;
        
        Mat outMat = m.clone();
        
        double[] bluePixel = new double[3];
        bluePixel[0]=MAX_PIXEL_VALUE;
        bluePixel[1]=MIN_PIXEL_VALUE;
        bluePixel[2]=MIN_PIXEL_VALUE;
        
        //********************************
        Mat tryMat = useMask(inversion(mask));
        //displayMatrix(tryMat, "rem Image");
        
        Mat iMat = inversion(tryMat);
        Mat newMask = new Mat(iMat.rows()+2, iMat.cols()+2, iMat.type());
        if(tryMat.get(1,1)[0]==MAX_PIXEL_VALUE){
            Imgproc.copyMakeBorder(iMat, newMask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);

            Imgproc.floodFill(iMat, newMask, new Point(1,1), new Scalar(MAX_PIXEL_VALUE/2), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);
        }
        if(tryMat.get(1,tryMat.width()-1)[0]==MAX_PIXEL_VALUE){
            Imgproc.copyMakeBorder(iMat, newMask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);

            Imgproc.floodFill(iMat, newMask, new Point(tryMat.width()-1,1), new Scalar(MAX_PIXEL_VALUE/2), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);
        }
        if(tryMat.get(tryMat.height()-1,1)[0]==MAX_PIXEL_VALUE){
            Imgproc.copyMakeBorder(iMat, newMask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);

            Imgproc.floodFill(iMat, newMask, new Point(1,tryMat.height()-1), new Scalar(MAX_PIXEL_VALUE/2), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);
        }
        if(tryMat.get(tryMat.height()-1,tryMat.width()-1)[0]==MAX_PIXEL_VALUE){
            Imgproc.copyMakeBorder(iMat, newMask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);

            Imgproc.floodFill(iMat, newMask, new Point(tryMat.width()-1,tryMat.height()-1), new Scalar(MAX_PIXEL_VALUE/2), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);
        }        
        
        for(int i=0; i<iMat.rows(); i++){
            for(int j=0; j<iMat.cols(); j++){
                if(iMat.get(i, j)[0]!=MAX_PIXEL_VALUE/2 && mask.get(i, j)[0]!=MAX_PIXEL_VALUE){
                    outMat.put(i, j, bluePixel);
                }
            }
        }
        //displayMatrix(outMat, "mask Image");
        
        return outMat;
    }
    
    /**
     * Set points from list lp in matrix m to MAX_PIXEL_VALUE<br>
     * Return null on 'm' == null or 'lp' == null
     * 
     * @param m input matrix
     * @param lp list of points which will be MAX_PIXEL_VALUE
     * @return Matrix without points from lp
     */
    private Mat removeCWS(Mat m, ArrayList<Point> lp){
        if(m==null || lp==null)
            return null;
        Mat outMat = m.clone();
        for(Point p : lp){
            outMat.put((int)p.x, (int)p.y, MAX_PIXEL_VALUE);
        }
        return outMat;
    }
    
    /**
     * Rescale matrix on power base<br>
     * Return null on 'm' == null or class variable 'maskMat' == null
     * 
     * @param m matrix for rescale
     * @return Rescaled matrix
     */
    private Mat powerRescale(Mat m){
        if(m==null || maskMat==null)
            return null;
        
        double pom;
        double min = 260;
        double max = -1;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pom = m.get(i, j)[0];
                    if(pom>max)
                        max=pom;
                    if(pom<min)
                        min=pom;
                }
            }
        }
        
        Mat outM = copyBlankMatrix(m);
        double pixelConst = (double)MAX_PIXEL_VALUE / Math.pow(max-min, 2);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pom = Math.pow(m.get(i, j)[0]-min, 2)*pixelConst;
                    outM.put(i, j, pom);
                }
                else
                    outM.put(i, j, MIN_PIXEL_VALUE);
            }
        }
        //displayMatrix(outM, "SQ");
        
        return outM;
    }
    
    /**
     * For remove veins connected to point in optical disc<br>
     * Return null on 'm'==null or 'p'==null
     * 
     * @param m matrix for flood
     * @param p seed point
     * @return Matrix without vein (flood from set point)
     */
    private Mat floodFromOD(Mat m, Point p){
        if(m==null || p==null)
            return null;
        Mat nobgMat = m.clone();
        Mat mask = new Mat(nobgMat.rows()+2, nobgMat.cols()+2, nobgMat.type());
        Imgproc.copyMakeBorder(nobgMat, mask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);
        
        nobgMat=inversion(nobgMat);
        Imgproc.floodFill(nobgMat, mask, new Point(p.y,p.x), new Scalar(0), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);
        nobgMat=inversion(nobgMat);
        nobgMat=useMask(nobgMat);
        Imgproc.medianBlur(nobgMat, nobgMat, 3);
        //displayMatrix(nobgMat, "flood Image");
        
        return nobgMat;
    }
    
    /**
     * Create approximal background from matrix m and remove every pixel which lies in range set by diff<br>
     * Return null on 'm'==null or class variable 'maskMat'==null
     * 
     * @param m matrix for background removal
     * @param diff value of how much can be pixels of image background and created background different
     * @return Matrix without background
     */
    private Mat removeBackground(Mat m, int diff){
        if(m==null || maskMat==null)
            return null;
        Mat bgMat = copyBlankMatrix(m);
        Imgproc.medianBlur(m, bgMat, 55);
        //displayMatrix(bgMat, "Background Image");
        
        //remove background
        double pom;
        Mat nobgMat = copyBlankMatrix(m);
        for(int i=0; i<bgMat.rows(); i++){
            for(int j=0; j<bgMat.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pom = bgMat.get(i, j)[0]-m.get(i, j)[0];
                    if(pom<diff && pom>-diff)
                        nobgMat.put(i, j, MAX_PIXEL_VALUE);
                    else
                        nobgMat.put(i, j, MIN_PIXEL_VALUE);
                }
                else{
                    nobgMat.put(i, j, MAX_PIXEL_VALUE);
                }
            }
        }
        //displayMatrix(nobgMat, "just vein Image");
        
        return nobgMat;
    }
    
    /**
     * Remove veins from matrix m depending on their length as line<br>
     * Return null on 'm'==null
     * 
     * @param m matrix with veins
     * @return Matrix without veins
     */
    private Mat veinRemove(Mat m){
        if(m==null)
            return null;
        
        Mat yMat = copyBlankMatrix(m);
        Imgproc.Sobel(m, yMat, CvType.CV_16U, 0, 2, 5, 1, 0);
        yMat.convertTo(yMat, CvType.CV_8UC3);
        //displayMatrix(yMat, "dy Image");
        
        Mat xMat = copyBlankMatrix(m);
        Imgproc.Sobel(m, xMat, CvType.CV_16U, 2, 0, 5, 1, 0);
        xMat.convertTo(xMat, CvType.CV_8UC3);
        //displayMatrix(xMat, "dx Image");
        
        //Gradient Image
        int d;
        Mat myMat = copyBlankMatrix(xMat);
        for(int i=0; i<xMat.rows(); i++){
            for(int j=0; j<xMat.cols(); j++){
                d=(int)((xMat.get(i, j)[0]+yMat.get(i, j)[0]));
                if(d>=MAX_PIXEL_VALUE)
                    d=MAX_PIXEL_VALUE;
                myMat.put(i, j, d);
            }
        }
        //Thresholded
        Imgproc.GaussianBlur(myMat, myMat, new Size(3,3), 0);        
        Imgproc.threshold(myMat, myMat, MIN_PIXEL_VALUE/2, MAX_PIXEL_VALUE, Imgproc.THRESH_OTSU);
        Imgproc.medianBlur(myMat, myMat, 5);
        //displayMatrix(myMat, "thresh Image");
        
        //find Veins
        int minLength=20;
        int maxGap=5;
        Mat lines = new Mat();
        Imgproc.HoughLinesP(myMat, lines, 3, Math.PI/180.0, 100, minLength, maxGap);        
        Mat linesMat = copyBlankMatrix(m);
        
        for (int x = 0; x < lines.cols(); x++){
            double[] vec = lines.get(0, x);
            double x1 = vec[0], 
                   y1 = vec[1],
                   x2 = vec[2],
                   y2 = vec[3];
            Core.line(linesMat, new Point(x1, y1), new Point(x2, y2), new Scalar(MAX_PIXEL_VALUE), 3);
        }
        //displayMatrix(linesMat, "Lines Image");
        
        return linesMat;
    }
    
    /**
     * Remove optical disc (the biggest bright spot) from matrix m and get let only others bright spots.<br>
     * Also fill class variable 'discMat' with matrix with located optical disc<br>
     * Return null on 'm'==null
     * 
     * @param m preprocessed matrix
     * @return Matrix with spots indicating sure white spots in retina image
     */
    private Mat opticDiscRecognition(Mat m){
        if(m==null)
            return null;
        
        Mat opticMat = inversion(m);
        Imgproc.threshold(opticMat, opticMat, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE, Imgproc.THRESH_OTSU);
        opticMat = inversion(useMask(inversion(opticMat)));
        Mat outMat = opticMat.clone();
        
        //find only points(contours) in nonblack area
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(opticMat, contours, new Mat(), Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_NONE);
        double maxSize=0;
        int index=-1;
        for (int i=0; i<contours.size(); i++) {
            if(Imgproc.contourArea(contours.get(i))>maxSize){
                maxSize=Imgproc.contourArea(contours.get(i));
                index=i;
            }
        }
        if(index!=-1){
            Rect rect = Imgproc.boundingRect(contours.get(index));
        
            int maxY,maxX,minY,minX;
        
            int biggerSide=rect.height;
            if(rect.width>biggerSide)
                biggerSide=rect.width;
        
            minX=rect.x;
            minY=rect.y;
            maxX=rect.x+rect.width;
            maxY=rect.y+rect.height;
            Point middle = new Point(rect.x+(int)(rect.width/2), rect.y+(int)(rect.height/2));
            discMat = copyBlankMatrix(m);
        
            //after testing I decide to make rectangle bigger on left/right(depend on position of OD), because biggest part lies on right/left of veins and so center is moved
            double magicConst=0.25;
            Point p1, p2;
            if(middle.x<=(m.cols()/2)){//on left
                p1 = new Point(middle.x-(biggerSide/2)*(1+magicConst), middle.y-biggerSide/2);
                p2 = new Point(middle.x+biggerSide/2, middle.y+biggerSide/2);
            }
            else{//on right
                p1 = new Point(middle.x-biggerSide/2, middle.y-biggerSide/2);
                p2 = new Point(middle.x+(biggerSide/2)*(1+magicConst), middle.y+biggerSide/2);
            }

            //it cover another part of contours (make intersection with another found parts
            Rect R;
            Rectangle mainRect = new Rectangle((int)p1.x,(int)p1.y,(int)(p2.x-p1.x),(int)(p2.y-p1.y));
            Rectangle pomRect;
            for (int i=0; i<contours.size(); i++) {
                if(i!=index){
                    R=Imgproc.boundingRect(contours.get(i));
                    pomRect = new Rectangle(R.x,R.y,R.width,R.height);
                    if(pomRect.intersects(mainRect)){
                        if(maxX<(R.x+R.width))
                            maxX=R.x+R.width;
                        if(maxY<(R.y+R.height))
                            maxY=R.y+R.height;
                        if(minX>(R.x))
                            minX=R.x;
                        if(minY>(R.y))
                            minY=R.y;
                    }
                }
            }
            Core.rectangle(discMat, new Point(minX,minY), new Point(maxX,maxY), new Scalar(MAX_PIXEL_VALUE), Core.FILLED);

            outMat=inversion(outMat);
            Core.rectangle(outMat, p1, p2, new Scalar(MAX_PIXEL_VALUE), Core.FILLED);
            return inversion(outMat);
        }
        else{
            discMat = copyBlankMatrix(m);
            return copyBlankMatrix(m);
        }
        
    }
    
    //New Optic Disc Localization Approach in Retinal Images
    //Not working at all !!
    /**
     * It should rescale pixel values so Optical Disc will be the best visible<br>
     * Return null on 'm' == null or class variable 'maskMat' == null<br>
     * 
     * Not working, don't use!!!
     * 
     * @param m Grayscaled image 
     * @return Matrix
     */
    private Mat opticalDisc(Mat m){
        if(m==null || maskMat == null)
            return null;
        
        Mat myM = new Mat(m.rows(), m.cols(),CvType.CV_32F);
        m.convertTo(myM, CvType.CV_32F);
        Mat opticMatL5xE5 = new Mat(m.rows(), m.cols(),CvType.CV_32F);
        Mat opticMatL5xS5 = new Mat(m.rows(), m.cols(),CvType.CV_32F);
        Mat opticMatE5xL5 = new Mat(m.rows(), m.cols(),CvType.CV_32F);
        Mat opticMatS5xL5 = new Mat(m.rows(), m.cols(),CvType.CV_32F);

        
        Mat L5xE5 = new Mat(new Size(5,5),CvType.CV_32F);
        addRow(L5xE5, 0, -1, -2, 0, 2, 1);
        addRow(L5xE5, 1, -4, -8, 0, 8, 4);
        addRow(L5xE5, 2, -6, -12, 0, 12, 6);
        addRow(L5xE5, 3, -4, -8, 0, 8, 4);
        addRow(L5xE5, 4, -1, -2, 0, 2, 1);
        Imgproc.filter2D(myM, opticMatL5xE5, -1, L5xE5);
        //displayMatrix(opticMatL5xE5, "L5xE5");
        
        Mat L5xS5 = new Mat(new Size(5,5),CvType.CV_32F);
        addRow(L5xS5, 0, -1, 0, 2, 0, 1);
        addRow(L5xS5, 1, -4, 0, 8, 0, 4);
        addRow(L5xS5, 2, -6, 0, 12, 0, 6);
        addRow(L5xS5, 3, -4, 0, 8, 0, 4);
        addRow(L5xS5, 4, -1, 0, 2, 0, 1);
        Imgproc.filter2D(myM, opticMatL5xS5, -1, L5xS5);
        //displayMatrix(opticMatL5xS5, "L5xS5");
        
        Mat E5xL5 = new Mat(new Size(5,5),CvType.CV_32F);
        addRow(E5xL5, 0, -1, -4, -6, -4, -1);
        addRow(E5xL5, 1, -2, -8, -12, -8, -2);
        addRow(E5xL5, 2, 0, 0, 0, 0, 0);
        addRow(E5xL5, 3, 2, 8, 12, 8, 2);
        addRow(E5xL5, 4, 1, 4, 6, 4, 1);
        Imgproc.filter2D(myM, opticMatE5xL5, -1, E5xL5);
        //displayMatrix(opticMatE5xL5, "E5xL5");
        
        Mat S5xL5 = new Mat(new Size(5,5),CvType.CV_32F);
        addRow(S5xL5, 0, -1, -4, -6, -4, -1);
        addRow(S5xL5, 1, 0, 0, 0, 0, 0);
        addRow(S5xL5, 2, 2, 8, 12, 8, 2);
        addRow(S5xL5, 3, 0, 0, 0, 0, 0);
        addRow(S5xL5, 4, -1, -4, -6, -4, -1);
        Imgproc.filter2D(myM, opticMatS5xL5, -1, S5xL5);
        //displayMatrix(opticMatS5xL5, "S5xL5");
        
        double[][] pomArray = new double[m.rows()][m.cols()];
        double f1, f2, f3, f4, f;
        double min=Double.MAX_VALUE;
        double max=-Double.MAX_VALUE;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                //if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    f1=opticMatL5xE5.get(i, j)[0];
                    f2=opticMatL5xS5.get(i, j)[0];
                    f3=opticMatE5xL5.get(i, j)[0];
                    f4=opticMatS5xL5.get(i, j)[0];
                    f=Math.sqrt(f1*f1+f2*f2+f3*f3+f4*f4);
                    pomArray[i][j]=f;
                    if(f<min)
                        min=f;
                    if(f>max)
                        max=f; 
                /*}
                else{
                    pomArray[i][j]=0;
                }*/
            }
        }
        //System.out.println(min + " " + max);
        
        Mat FMat = copyBlankMatrix(m);
        double constPixel = MAX_PIXEL_VALUE/(max-min);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    FMat.put(i, j, ((pomArray[i][j]-min)*constPixel));
                }
                else{
                    FMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        
        double almostMax = 0.7*(max-min)*constPixel;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE && FMat.get(i, j)[0]>almostMax){//inside mask and under procentual threshold
                    FMat.put(i, j, MAX_PIXEL_VALUE);
                }
                else{
                    FMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        
        return FMat;
    }
    
    /**
     * Add row to matrix with 5 columns<br>
     * Return without adding when cols or row is not matching
     * 
     * @param m Mat to which will be row added
     * @param row number of added row
     * @param x1 1. number
     * @param x2 2. number
     * @param x3 3. number 
     * @param x4 4. number
     * @param x5 5. number
     */
    private void addRow(Mat m, int row, double x1, double x2, double x3, double x4, double x5){
        if(m.cols()!=5 || m.rows()>row)
            return;
        m.put(row, 0, x1); m.put(row, 1, x2); m.put(row, 2, x3); m.put(row, 3, x4); m.put(row, 4, x5);
    }
    
    /**
     * Create matrix of original resized image with green places covering hermorrhages <br>
     * Return null on 'detected' == null or class variable 'imageMat' == null or class variable 'discMat' == null
     * 
     * @param detected Mat of detected places
     * @param sureArea Mat of detected places (bigger places but not all)
     * @return Matrix with RGB image
     */
    private Mat CWS(Mat detected, Mat sureArea){
        if(detected==null || imageMat==null || discMat==null || sureArea==null)
            return null;
        
        double[] greenPixel = new double[3];
        greenPixel[0]=MIN_PIXEL_VALUE;
        greenPixel[1]=MAX_PIXEL_VALUE;
        greenPixel[2]=MIN_PIXEL_VALUE;
        
        
        Mat out = copyBlankMatrix(imageMat);
        ArrayList<Point> lp = new ArrayList<>();
        for(int i=0; i<imageMat.rows(); i++){
            for(int j=0; j<imageMat.cols(); j++){
                if(detected.get(i, j)[0]==MIN_PIXEL_VALUE && discMat.get(i, j)[0]!=MAX_PIXEL_VALUE){
                    out.put(i, j, greenPixel);
                    lp.add(new Point(i,j));
                }
                else
                    out.put(i, j, imageMat.get(i, j));
            }
        }
        Mat mask = new Mat(sureArea.rows()+2, sureArea.cols()+2, sureArea.type());
        Imgproc.copyMakeBorder(sureArea, mask, 1, 1, 1, 1, Imgproc.BORDER_REPLICATE);
        
        mask=inversion(mask);
        
        for(Point p: lp){
            if(sureArea.get((int)p.x, (int)p.y)[0]==MAX_PIXEL_VALUE){
                Imgproc.floodFill(sureArea, mask, new Point((int)p.y, (int)p.x), new Scalar(100), null, new Scalar(0), new Scalar(0), Imgproc.FLOODFILL_FIXED_RANGE);

            }
            
        }
        //displayMatrix(sureArea, "Found things Image");
        for(int i=0; i<imageMat.rows(); i++){
            for(int j=0; j<imageMat.cols(); j++){
                if(sureArea.get(i, j)[0]==100){
                    out.put(i, j, greenPixel);
                }
            }
        }
        
        return out;
    }
    
    /**
     * Find all green spots (detected CWS) in matrix<br>
     * Return null on 'm'==null
     * 
     * @param m matrix with green detected CWS
     * @return list of CWS points
     */
    private ArrayList<Point> getCWSPoints(Mat m){
        if(m==null)
            return null;
        
        double[] greenPixel = new double[3];
        greenPixel[0]=MIN_PIXEL_VALUE;
        greenPixel[1]=MAX_PIXEL_VALUE;
        greenPixel[2]=MIN_PIXEL_VALUE;
        
        ArrayList<Point> lp = new ArrayList<>();
        
        for(int i=0; i<imageMat.rows(); i++){
            for(int j=0; j<imageMat.cols(); j++){
                if(m.get(i, j)[0]==greenPixel[0] && m.get(i, j)[1]==greenPixel[1] && m.get(i, j)[2]==greenPixel[2]){
                    lp.add(new Point(i,j));
                }
            }
        }
        
        return lp;
    }
    
    /**
     * Only for recursive function: private Mat floodFill()
     * 
     * @param m matrix
     * @param p seed point
     * @param deviation accepted difference
     */
    private void recFlood(Mat m, Point p, int deviation){
        if(m.get((int)p.x, (int)p.y)[0]>=(MAX_PIXEL_VALUE-deviation)){
            m.put((int)p.x, (int)p.y, 128);
            recFlood(m, new Point((int)p.x-1, (int)p.y-1), deviation);
            recFlood(m, new Point((int)p.x-1, (int)p.y), deviation);
            recFlood(m, new Point((int)p.x-1, (int)p.y+1), deviation);
            //------
            recFlood(m, new Point((int)p.x, (int)p.y-1), deviation);
            recFlood(m, new Point((int)p.x, (int)p.y+1), deviation);
            //------
            recFlood(m, new Point((int)p.x+1, (int)p.y-1), deviation);
            recFlood(m, new Point((int)p.x+1, (int)p.y), deviation);
            recFlood(m, new Point((int)p.x+1, (int)p.y+1), deviation);
        }
    }
    
    //is not save to use!!
    /**
     * worst alternative of cv.floodFill(); not recomended to use because of stack overflow!!!!
     */
    private Mat floodFill(Mat m, ArrayList<Point> L, int deviation){
        if(m==null || L==null)
            return null;
        
        Mat myMat = m.clone();
        
        for(Point p : L)
            recFlood(myMat, p, deviation);
        
        return myMat;
    }
    
    /**
     * Create matrix by joining two matrix and MIN_PIXEL_VALUE indicates diseased places <br>
     * Return null on 'clahe' == null or 'potential' == null or class variable 'maskMat' == null
     * 
     * @param clahe Matrix after CLAHE algorithm
     * @param potential Matrix with potential diseases
     * @return Matrix of maximal potential diseased places
     */
    private Mat maxArea(Mat clahe, Mat potential){
        if(clahe==null || potential==null || maskMat==null)
            return null;
        
        Mat maxMat = copyBlankMatrix(clahe);
        int pom;
        for(int i=0; i<clahe.rows(); i++){
            for(int j=0; j<clahe.cols(); j++){
                pom=(int)(clahe.get(i, j)[0]+potential.get(i, j)[0]);
                if(pom>MAX_PIXEL_VALUE)
                    maxMat.put(i,j,MAX_PIXEL_VALUE);
                if(pom<MIN_PIXEL_VALUE)
                    maxMat.put(i, j, MIN_PIXEL_VALUE);
                if(pom>=MIN_PIXEL_VALUE && pom<=MAX_PIXEL_VALUE)
                    maxMat.put(i, j, pom);
            }
        }
        maxMat=useMask(maxMat);
        for(int i=0; i<clahe.rows(); i++){
            for(int j=0; j<clahe.cols(); j++){
                if(maxMat.get(i, j)[0]!=MAX_PIXEL_VALUE)
                    maxMat.put(i, j, MIN_PIXEL_VALUE);
            }
        }
        
        return maxMat;
    }
    
    /**
     * Find pixels with Von Neumann neighborhood 1 after adaptive threshold <br>
     * Return null on 'm' == null or class variable 'maskMat' == null
     * 
     * @param m Matrix with potential diseases
     * @return Matrix of sure places with disease
     */
    private Mat minArea(Mat m){
        if(m==null || maskMat==null)
            return null;
        
        Mat mat = copyBlankMatrix(m);
        Imgproc.GaussianBlur(m, mat, new Size(3,3), 0);
        Imgproc.adaptiveThreshold(mat, mat, MAX_PIXEL_VALUE, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,  Imgproc.THRESH_BINARY, 3, 2);
        mat=useMask(mat);
        //displayMatrix(mat, "Thresh Image");
        
        int filterSize=3;
        Mat element = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_CROSS, new Size(filterSize,filterSize));
        //morphological close
        Imgproc.morphologyEx(mat, mat, Imgproc.MORPH_CLOSE, element);
        //displayMatrix(mat, "Cross Image");
        
        Mat pointMat = copyBlankMatrix(m);
        int armOfCross = filterSize/2;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE && j>=armOfCross && j<(m.cols()-armOfCross) && i>=armOfCross && i<(m.rows()-armOfCross)){//inside mask and in range of cross
                    if(mat.get(i, j)[0]==MIN_PIXEL_VALUE && mat.get(i-1, j)[0]==MIN_PIXEL_VALUE && mat.get(i+1, j)[0]==MIN_PIXEL_VALUE && mat.get(i, j-1)[0]==MIN_PIXEL_VALUE && mat.get(i, j+1)[0]==MIN_PIXEL_VALUE)
                        pointMat.put(i, j, MIN_PIXEL_VALUE);
                    else
                        pointMat.put(i, j, MAX_PIXEL_VALUE);
                }
                else{
                    pointMat.put(i, j, MAX_PIXEL_VALUE);
                }
            }
        }
        //displayMatrix(pointMat, "Point Image");
        
        int aroundPoint=5;
        int threshold=3;
        int counter;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE && j>=aroundPoint && j<(m.cols()-aroundPoint) && i>=aroundPoint && i<(m.rows()-aroundPoint) && pointMat.get(i, j)[0]==MIN_PIXEL_VALUE){//inside mask and around point and point is black
                    //search around point
                    counter=0;
                    for(int k=i-aroundPoint; k<i+aroundPoint; k++){
                        for(int l=j-aroundPoint; l<j+aroundPoint; l++){
                            if(pointMat.get(k, l)[0]==MIN_PIXEL_VALUE){
                                counter++;
                            }
                        }    
                    }
                    if(counter>=threshold){
                        mat.put(i, j, MIN_PIXEL_VALUE);
                    }
                    else{
                        mat.put(i, j, MAX_PIXEL_VALUE);
                    }
                }
                else{
                    mat.put(i, j, MAX_PIXEL_VALUE);
                }
            }
        }
        
        return mat;
    }
    
    /**
     * Estimate background which gets rid of vessels and then rescale matrix to MIN_PIXEL_VALUE-MAX_PIXEL_VALUE<br>
     * !!! This method change class variable 'maskMat' !!!<br>
     * Return null on 'm' == null or class variable 'maskMat' == null
     * 
     * @param m Preprocessed matrix (the best is CLAHE)
     * @return Matrix with potential diseases
     */
    private Mat potentialCWS(Mat m){
        if(m==null || maskMat==null)
            return null;
        
        double pomPixel;        

        //blur to estimate background
        Mat medianBlurMat = copyBlankMatrix(m);
        Imgproc.medianBlur(m, medianBlurMat, 55);
        //displayMatrix(medianBlurMat, "Blurred Image");
//*****************************
//******sub blurred from CLAHE to find markants (vessels)       
        Mat backMat = copyBlankMatrix(m);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pomPixel=m.get(i, j)[0] - medianBlurMat.get(i, j)[0];
                    if(pomPixel<MIN_PIXEL_VALUE)
                        backMat.put(i, j, MIN_PIXEL_VALUE);
                    else
                        backMat.put(i, j, pomPixel);
                }
                else{
                    backMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        //displayMatrix(backMat, "Background Image");
//*****************************
//******create new mask + remove vessels (by mask fake)
        Mat subMat = copyBlankMatrix(m);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pomPixel=m.get(i, j)[0] - backMat.get(i, j)[0];
                    if(pomPixel<MIN_PIXEL_VALUE)
                        subMat.put(i, j, MIN_PIXEL_VALUE);
                    else
                        subMat.put(i, j, pomPixel);
                }
                else{
                    subMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        createMask(subMat);
//*****************************
//******change to MIN_PIXEL_VALUE-MAX_PIXEL_VALUE scale
        double max=-Double.MAX_VALUE;
        double min=Double.MAX_VALUE;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pomPixel = subMat.get(i, j)[0];
                    if(pomPixel>max)
                        max=pomPixel;
                    if(pomPixel<min)
                        min=pomPixel;
                }
            }
        }
        double pixelConst = MAX_PIXEL_VALUE/(max-min);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    pomPixel = subMat.get(i, j)[0];
                    subMat.put(i, j, (pomPixel-min)*pixelConst);
                }
            }
        }
//*****************************   
        //displayMatrix(subMat, "pot CWS");
        return subMat;
    }
    
    /**
     * Convert matrix to CLAHE with param. (2.0,8,8)<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for CLAHE algorithm
     * @return Matrix after CLAHE algorithm
     */
    private Mat CLAHE(Mat m){
        if(m==null)
            return null;
        
        CLAHE clahe = new CLAHE(2.0,8,8);
        return clahe.apply(inversion(m));
    }
    
    //http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
    /**
     * Create x+y gradient matrix of m<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for gradient calculation
     * @return Matrix with x & y gradient
     */
    private Mat gradient(Mat m){
        if(m==null)
            return null;

        Mat grad_x = copyBlankMatrix(m);
        Imgproc.Sobel(m, grad_x, CvType.CV_16S, 1, 0);
        Mat grad_y = copyBlankMatrix(m);
        Imgproc.Sobel(m, grad_y, CvType.CV_16S, 0, 1);

        Mat abs_grad_x = copyBlankMatrix(m);
        grad_x.convertTo(abs_grad_x, CvType.CV_8UC3);
        Mat abs_grad_y = copyBlankMatrix(m);
        grad_y.convertTo(abs_grad_y, CvType.CV_8UC3);

        Mat outGradMat = copyBlankMatrix(m);
        double d;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                d=(abs_grad_x.get(i, j)[0]+abs_grad_y.get(i, j)[0]);
                if(d>=MAX_PIXEL_VALUE)
                    d=MAX_PIXEL_VALUE;
                outGradMat.put(i, j, d);
            }
        }
        
        return outGradMat;
    }
    
    /**
     * Invert values of matrix (MIN_PIXEL_VALUE-MAX_PIXEL_VALUE -> MAX_PIXEL_VALUE-MIN_PIXEL_VALUE)<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for inversion
     * @return Inverted matrix
     */
    private Mat inversion(Mat m){
        if(m==null)
            return null;
        Mat mo = copyBlankMatrix(m);
        
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                mo.put(i, j, (MAX_PIXEL_VALUE-m.get(i, j)[0]));
            }
        }
        
        return mo;
    }
    
    /**
     * Old funcion which try to threshold image with histogram equalization. From first paper which I tried<br>
     * Not recomended for use!!
     * 
     * @param preprocMat input preprocessed matrix
     */
    private void retinalHemorrhages(Mat preprocMat){
        //Non uniform illumination ???
        //easy way - EqualizeHist
        Mat histoMat = histogramEqualization(preprocMat);
        //Imgproc.equalizeHist(preprocMat, histoMat);
        displayMatrix(histoMat, "Illumination normalization");

        //local contrast variability remove
        //Contrast = max(pixel intensity of area)- min(pixel intensity of area)

        //adaptive threshold
        Mat adaptedMat = copyBlankMatrix(preprocMat);
        Imgproc.adaptiveThreshold(histoMat, adaptedMat, MAX_PIXEL_VALUE, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,  Imgproc.THRESH_BINARY, 11, -5);
        //adaptedMat = RAT(histoMat,8);
        displayMatrix(adaptedMat, "Adapted Thresh");
    }
    
    //BIO str. 41
    /**
     * RAT algorithm for create average value in moving window<br>
     * Return null on 'm' == null or matrix smaller then size or when size is odd number
     * 
     * @param m Input matrix
     * @param size Even size of moving window (usually 8)
     * @return Matrix after image averaging
     */
    private Mat RAT(Mat m, int size){
        if(m==null || m.rows()<size || m.cols()<size || size%2!=0)
            return null;
        Mat RATMat = m.clone();
        
        double sum=0;
        int avg;
        
        for(int i=0; i<(m.rows()-size); i+=size){
            for(int j=0; j<(m.cols()-size); j+=(size/2)){
                for(int k=0; k<size; k++){
                    for(int l=0; l<size; l++){
                        sum+=m.get(i+k, j+l)[0];
                    }
                }
                avg=(int)(sum/(size*size));
                sum=0;
                for(int k=0; k<size; k++){
                    for(int l=0; l<(size/2); l++){
                        RATMat.put(i+k, j+l, avg);
                    }
                }
            }
        }
        
        return RATMat;
    }
    
    //http://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
    /**
     * Change pixel intensities so all values have same count in image (only retina vithout border)<br>
     * Return null on 'm' == null or class variable 'maskMat' == null
     * 
     * @param m Matrix for equalization
     * @return Equalized matrix
     */
    private Mat histogramEqualization(Mat m){
        if(m==null || maskMat==null)
            return null;
        
        int Range=256;
        double[] p = new double[Range];
        int numberOfPixels=0;
        
        for(int i=0; i<Range; i++)
            p[i]=0;
        
        //count of intensities
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    p[(int)m.get(i, j)[0]]++;
                    numberOfPixels++;
                }
            }
        }
        
        for(int i=0; i<Range; i++)
            p[i]=p[i]/numberOfPixels;
        
        //equalized image
        double[] pIntegrated = new double[Range];
        for(int i=0; i<Range; i++){
            pIntegrated[i]=0;
            for(int j=0; j<i; j++){
                pIntegrated[i]+=p[j];
            }
            pIntegrated[i]=Math.floor((Range-1)*pIntegrated[i]);
        }
        
        Mat equMat = copyBlankMatrix(m);
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]!=MAX_PIXEL_VALUE){//inside mask
                    equMat.put(i, j, pIntegrated[(int)m.get(i, j)[0]]);
                }
                else{
                    equMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        
        return equMat;
    }
    
    /**
     * Extract H/S/I channel from original resized image matrix<br>
     * Return null on class variable 'imageMat' == null
     * 
     * @param channel Choose channel which will be returned - 'H' / 'S' / 'I'
     * @return Matrix of H/S/I
     */
    private Mat getHSI(char channel){
        if(imageMat==null)
            return null;
                
        
        Mat HSIMat = HSI(imageMat);
        //displayMatrix(HSIMat, "HSI");
        ArrayList<Mat> hsi = new ArrayList<>();
        split(HSIMat,hsi);
        switch(Character.toUpperCase(channel)){
            case 'H':
                return hsi.get(0);
            case 'S':
                return hsi.get(1);
            case 'I':
                return hsi.get(2);
            default:
                return null;
        }
    }
    
    /**
     * From original image is separated green channel, remove noise and then is created and used mask<br>
     * Return null on class variable 'imageMat' == null
     * 
     * @return Matrix with preprocessed 'imageMat'
     */
    private Mat preprocessing(){
        if(imageMat==null)
            return null;
        
        //only Green channel
        ArrayList<Mat> rgb = new ArrayList<>();
        split(imageMat,rgb);
        Mat imageGrayMat = rgb.get(1);  
        //displayMatrix(imageGrayMat, "Green channel");
                
        //smooth noise    
        Mat medianBlurMat = copyBlankMatrix(imageGrayMat);
        Imgproc.medianBlur(imageGrayMat, medianBlurMat, 3);
        //displayMatrix(medianBlurMat, "Blurred");
        
        //background mask
        createMask(imageGrayMat);
        //displayMatrix(maskedMat, "Background mask");
        
        //apply mask
        Mat maskedMat = useMask(medianBlurMat);
        //displayMatrix(maskedMat, "Used mask");
        
        //Bilinear interpolation???
        return maskedMat;
    }
    
    /**
     * Create mask for remove retina border saved in class variable 'maskMat'. Threshold value is MASK_THRESHOLD<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix from which is mask created
     */
    private void createMask(Mat m){
        if(m==null)
            return;
        maskMat = copyBlankMatrix(m);
        
        //split by threshold
        Imgproc.threshold(m, maskMat, MASK_THRESHOLD, MAX_PIXEL_VALUE, Imgproc.THRESH_BINARY_INV);
        
        int filterSize=3;
        Mat element = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(filterSize,filterSize));
        //morphological open & close
        Imgproc.morphologyEx(maskMat, maskMat, Imgproc.MORPH_OPEN, element);
        Imgproc.morphologyEx(maskMat, maskMat, Imgproc.MORPH_CLOSE, element);
        //displayMatrix(maskMat, "mask");
    }
    
    /**
     * Use 'maskMat' on chosen matrix.<br>
     * Return null on class variable 'maskMat' == null or 'm' == null or non matching size of those two matrices
     * 
     * @param m Matrix which will be masked
     * @return Matrix with used mask
     */
    private Mat useMask(Mat m){
        if(maskMat==null || m==null || m.rows()!=maskMat.rows() || m.cols()!=maskMat.cols())
            return null;
        
        Mat maskedMat = m.clone();
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                if(maskMat.get(i, j)[0]==MAX_PIXEL_VALUE)
                    maskedMat.put(i, j, maskMat.get(i, j));
            }
        }
        return maskedMat;
    }
    
    //https://en.wikipedia.org/wiki/HSL_and_HSV
    /**
     * Compute light model from RGB matrix.<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for lightening in RGB
     * @return Matrix with light image
     */
    private Mat getLight(Mat m){
        if(m==null)
            return null;
        
        ArrayList<Mat> rgb = new ArrayList<>();
        split(m,rgb); 
        
        Mat L = rgb.get(1);
        
        double R,G,B,min,max;
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                R=m.get(i, j)[0];
                G=m.get(i, j)[1];
                B=m.get(i, j)[2];
                
                min=Double.min(R,Double.min(G,B));
                max=Double.max(R,Double.max(G,B));
                
                L.put(i, j, 0.5*(min+max));
            }
        }
        
        return L;
    }
    
    //from: http://www.had2know.com/technology/hsi-rgb-color-converter-equations.html
    /**
     * Compute HSI model from RGB matrix.<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for HSI in RGB
     * @return HSI matrix
     */
    private Mat HSI(Mat m){
        if(m==null)
            return null;
        
        Mat HSI = copyBlankMatrix(m);
        double R,G,B,H,S,I,L,min,max;
        double[] mHSI = new double[3];
        
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                R=m.get(i, j)[0];
                G=m.get(i, j)[1];
                B=m.get(i, j)[2];
                
                I=(R+G+B)/3;
                
                if(I==0){
                    S=0;
                }
                else{
                    min=Double.min(R,Double.min(G,B));
                    S=1-(min/I);
                }
                
                H = Math.acos((R-0.5*G-0.5*B)/Math.sqrt(R*R+G*G+B*B-R*G-R*B-G*B));
                if(B>G)
                    H=((360*Math.PI)/180)-H;
                
                
                mHSI[0]=H*180/Math.PI;
                mHSI[1]=S*MAX_PIXEL_VALUE;
                mHSI[2]=I;
                HSI.put(i, j, mHSI);
            }
        }
        
        return HSI;
    }
    
    /**
     * Convert image image to matrix<br>
     * Return null on 'bi' == null
     * 
     * @param bi Image for conversion
     * @return Mat with data from bi
     */
    private Mat bufferedImageToMat(BufferedImage bi) {
        if(bi==null)
            return null;
        
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }
    
    /**
     * Convert matrix to image<br>
     * Return null on 'm' == null
     * 
     * @param m Matrix for conversion
     * @return Image with data from m
     */
    private BufferedImage matToBufferedImage(Mat m){
        if(m==null)
            return null;
        
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if ( m.channels() > 1 )
            type = BufferedImage.TYPE_3BYTE_BGR;
        
        BufferedImage bi = new BufferedImage(m.cols(),m.rows(), type);
        byte[] data = new byte[m.rows() * m.cols() * m.channels()];
        m.get(0, 0, data);
        byte[] imgPixels = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        System.arraycopy(data, 0, imgPixels, 0, data.length);
        return bi;
    }
    
    /**
     * Display matrix in window
     * 
     * @param m Matrix which will be dislayed
     * @param name Name of window
     */
    private void displayMatrix(Mat m, String name){
        displayImage(matToBufferedImage(m),name);
    }
    
    /**
     * Copy size and type of matrix and create new filled with MIN_PIXEL_VALUE
     * 
     * @param m Matrix which parameters will be used
     * @return New blank matrix
     */
    private Mat copyBlankMatrix(Mat m){
        Mat myMat = new Mat(m.height(), m.width(), m.type());
        if(m.type()==CvType.CV_8U){
            for(int i=0; i<m.rows(); i++){
                for(int j=0; j<m.cols(); j++){
                    myMat.put(i, j, MIN_PIXEL_VALUE);
                }
            }
        }
        return myMat;
    }
    
    /**
     * Show window with image and name with option 'Save image' on right mouse click<br>
     * Return null on 'img' == null
     * 
     * @param img Image which will be dislayed
     * @param name Name of window
     */
    private void displayImage(BufferedImage img, String name){   
        if(img==null){
            System.out.println("Failed to process image");
            System.exit(0);
        }
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame(name);
        frame.setLayout(new FlowLayout());        
        frame.setSize(img.getWidth()+30, img.getHeight()+40);     
        JLabel l=new JLabel();
        l.setIcon(icon);
        
        l.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent me){
                switch(me.getModifiers()){
                    case 4: //right mouse button
                        JFileChooser fileChooser = new JFileChooser();
                        fileChooser.setCurrentDirectory(new File(System.getProperty("user.home")));
                        int result = fileChooser.showSaveDialog(frame);
                        if (result == JFileChooser.APPROVE_OPTION) {
                            String path = fileChooser.getSelectedFile().getAbsolutePath();
                            try{
                                File outputfile = new File(path+".png");
                                ImageIO.write(img, "png", outputfile);
                                JOptionPane.showMessageDialog(null, "Success!\nImage was saved.", "Success", JOptionPane.INFORMATION_MESSAGE);
                            }catch (IOException e){
                                JOptionPane.showMessageDialog(null, "Error!\nImage was not saved.", "Error", JOptionPane.ERROR_MESSAGE);
                            }
                        }
                        break;
                    default:
                        break;
                }
            }            
        });
        
        frame.add(l);
        frame.setVisible(true);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
}