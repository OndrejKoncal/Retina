/**
 * @author Ondřej Končal
 * @date 29.11.2016
 * 
 * This program try to find diseases on retina.
 * Green one are CWS, blue one are hemorrhages
 * 
 * This program using external library/function CLAHE (author Michael Niephaus & Maurice Betzel)
 */
package retina;

import java.io.File;
import javax.swing.JFileChooser;
import javax.swing.JFrame;

public class Retina {    
    
    public static void main(String[] args) {
        /*String path = System.getProperty("user.dir")+"\\db\\images\\diaretdb1_image014.png";
        ImgCV ICV = new ImgCV(path);
        ICV.findDiseases();*/
        
        JFrame frame=new JFrame();
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showOpenDialog(frame);
        if (result == JFileChooser.APPROVE_OPTION) {
            String path = fileChooser.getSelectedFile().getAbsolutePath();
            ImgCV ICV = new ImgCV(path);
            ICV.findDiseases();
        }
        else{
            System.exit(0);
        }
    }
}
