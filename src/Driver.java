import static org.bytedeco.javacpp.opencv_core.Mat;
import static org.bytedeco.javacpp.opencv_core.MatVector;
import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;


/*
* This a demonstration-only to train a classifier with some images and ask
* that to predict an image.
*
* :NOTE:
* Before executing this class, please remove the LearnedData.yml.
* */
public class Driver {
    private final FaceRecognizerWrapper recognizerWrapper = new
            FaceRecognizerWrapper
            ("LearnedData.yml");

    public static void main(String args[]) {
//        String IMAGE = "/Users/snagaram/Pictures/Photos/group" +
//                ".jpg";
//        FaceDetectorWrapper detector = new FaceDetectorWrapper();
//        IplImage img = cvLoadImage(IMAGE);
//        Mat image = imread(IMAGE, IMREAD_GRAYSCALE);
//        MatVector faces = detector.detect(image);
////        System.out.println(faces.());
        Driver driver = new Driver();

        //Delete LearnedData.yml before executing train.
        driver.train();

        driver.predict();
    }

    private void predict() {
        FaceDetectorWrapper detector = new FaceDetectorWrapper();

        MatVector faces;
        Mat image = imread("./resources/msubject03happy.png", IMREAD_GRAYSCALE);
        faces = detector.detect(image);
//        System.out.println("hell"+faces.size());
//        recognizerWrapper.predict(faces, new Mat(new int[]{i + 1}));
        int[] predicted = recognizerWrapper.predict(faces);
        for (int i = 0; i < predicted.length; ++i) {
            System.out.println((i + 1) + " predicted as " + predicted[i]);
        }
//        System.out.println(predicted.toString());
    }

    public void train() {

        FaceDetectorWrapper detector = new FaceDetectorWrapper();

        MatVector faces;
//        = detector.detect(image);

        String files[] = {"msubject01glasses.png",
                "msubject02glasses.png",
                "msubject03glasses.png",
                "msubject04glasses.png",
                "msubject05glasses.png",
                "msubject06glasses.png"};

//        Mat image = imread("subject01glasses.gif", IMREAD_GRAYSCALE);
//        imwrite("dummy.gif",image);
        for (int i = 0; i < files.length; ++i) {
            Mat image = imread("./resources/" + files[i], IMREAD_GRAYSCALE);
//            imwrite("dummy.png",image);
//            System.out.println(image.size());

            faces = detector.detect(image);
            System.out.println(faces.size());
            recognizerWrapper.train(faces, new Mat(new int[]{i + 1}));
        }
        recognizerWrapper.close();
    }

}
