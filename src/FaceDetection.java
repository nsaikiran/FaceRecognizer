import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class FaceDetection {

    public static final String XML_FILE =
            //haarcascade.xml from opencv 2.4.0
            "/Users/snagaram/WorkSpace/extra/haarcascade.xml";
    public static final String IMAGE = "/Users/snagaram/Pictures/Photos/group" +
            ".jpg";

    public static void main(String[] args) {

        IplImage img = cvLoadImage(IMAGE);
        Mat original = imread(IMAGE);
        Mat grayscale = imread(IMAGE,IMREAD_GRAYSCALE);
        imwrite("orginal_written.jpg",original);
        imwrite("gray_written.jpg",grayscale);
//        detect(img);
    }

    public static void detect(IplImage src) {
        System.out.println(XML_FILE);
        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);

        // We can "cast" Pointer objects by instantiating a new object of the desired class.
        CvHaarClassifierCascade classifier = new CvHaarClassifierCascade
                (cvLoad(XML_FILE));
        if (classifier.isNull()) {
            System.err.println("Error loading classifier file \"" + XML_FILE + "\".");
            System.exit(1);
        }

        Pointer file = cvLoad(XML_FILE);
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(file);
        CvMemStorage storage = CvMemStorage.create();
        CvSeq sign = cvHaarDetectObjects(
                src,
                cascade,
                storage,
                1.1,
                3,
                CV_HAAR_DO_CANNY_PRUNING);
//        System.exit(0);

        cvClearMemStorage(storage);

        int total_Faces = sign.total();

        System.out.println("Total faces " + total_Faces);

        for (int i = 0; i < total_Faces; i++) {
            CvRect r = new CvRect(cvGetSeqElem(sign, i));
            cvSetImageROI(src,r);
//            IplImage cropped = cvCreateImage(cvGetSize(src), src.depth(), src
//                    .nChannels());
//            cvCreateImage(cvGetSize(src), src.depth(), src
//                    .nChannels());

//            cvCopy(src, cropped);
//            cvSaveImage("cropped.png", cropped);
            cvSaveImage("original"+i+".png",src);
//            cvRectangle(
//                    src,
//                    cvPoint(r.x(), r.y()),
//                    cvPoint(r.width() + r.x(), r.height() + r.y()),
//                    CvScalar.RED,
//                    2,
//                    LINE_AA,
//                    0);

        }
        System.out.println("I'm here");
//        cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);
//        cvShowImage("Result", src);
//        cvWaitKey(0);
        cvSaveImage("new.jpg",src);
        System.out.println("end");

//        cvSetImageROI(src,new CvRect(sign.x));
//        cvDestroyAllWindows();
//
    }
}