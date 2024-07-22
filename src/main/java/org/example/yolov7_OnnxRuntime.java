package org.example;

import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class yolov7_OnnxRuntime implements yolov7_OnnxRuntimeImpl{

    public static void main(String[] args) throws IOException, OrtException {
        yolov7_OnnxRuntime test = new yolov7_OnnxRuntime();
        test.readModel("src\\main\\resources\\model\\yolov7-tiny.onnx","person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush");

        test.Detect(0.7f, ImageIO.read(new File("src\\main\\resources\\image\\dog.jpg")));

    }


    private OrtEnvironment m_env;
    private OrtSession m_session;


    private ArrayList<String> m_classes = new ArrayList<>();
    private int model_input_width,model_input_height;
    private int resizedWidth,resizedHeight;
    private int offsetX, offsetY;

    @Override
    public boolean readModel(String onnxPath, String... classes) {
        try {
            // 加载ONNX模型
            m_env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            // 使用gpu,需要本机按钻过cuda，并修改pom.xml，不安装也能运行本程序   // sessionOptions.addCUDA(0);
            m_session = m_env.createSession(onnxPath, sessionOptions);
            // 输出基本信息
            Set<String> inputNames = m_session.getInputNames(); //images
            if (!inputNames.isEmpty()){
                //取第一个结果
                String inputName = inputNames.iterator().next();
                TensorInfo tensorInfo = (TensorInfo) m_session.getInputInfo().get(inputName).getInfo();

                long [] inputDims = tensorInfo.getShape();
                model_input_width = (int) inputDims[2];
                model_input_height = (int) inputDims[3];

                System.out.println(inputName + " " + model_input_width + "  " + model_input_height);


                for (String c : classes) {
                    m_classes.add(c);
                }

            }else{
                throw new Exception("获取输入信息失败");
            }

        }catch (Exception e){
            e.printStackTrace();
            return false;
        }
        return true;
    }

    @Override
    public ArrayList<OutResult> Detect(float minConfidence, BufferedImage image) throws OrtException {
        ArrayList<OutResult> resultArrayList = new ArrayList<>();


        float[][][] inputData = img2Float(image);

        OnnxTensor inputTensor = OnnxTensor.createTensor(m_env, new float[][][][]{inputData});
        Map<String, OnnxTensor> container = new HashMap<>();
        container.put("images", inputTensor);
        OrtSession.Result results = m_session.run(container);
        float[][] outputData = (float[][]) results.get(0).getValue();


        for (int i = 0 ; i < outputData.length ; i++){
            float x1 = outputData[i][1];
            float y1 = outputData[i][2];
            float x2 = outputData[i][3];
            float y2 = outputData[i][4];
            int classIdx = (int) outputData[i][5];
            float confidence = outputData[i][6];

            x1 = ((x1- offsetX) / resizedWidth) * image.getWidth();
            x2 = ((x2 - offsetX) / resizedWidth) * image.getWidth();
            y1 = ((y1 - offsetY) / resizedHeight) * image.getHeight();
            y2 = ((y2 - offsetY) / resizedHeight) * image.getHeight();

            resultArrayList.add(new OutResult((int)x1,(int)x2,(int)y1,(int)y2,classIdx,confidence));
        }

        for (OutResult r : resultArrayList){
            if (r.confidence > minConfidence){
                System.out.println("x1:"+r.x1+" y1:"+r.y1+" x2:"+r.x2+" y2:"+r.y2+" id:"+r.classId + " name:" + m_classes.get(r.classId) +" confidence:"+r.confidence);
            }
        }


        return resultArrayList;
    }

    @Override
    public float[][][] img2Float(BufferedImage image) {
        int cols = image.getWidth();
        int rows = image.getHeight();

        double scale;
        if ( cols / Double.valueOf(rows)  >  model_input_width / Double.valueOf(model_input_height) ){
            scale = model_input_width / Double.valueOf(cols);
        }else {
            scale = model_input_height / Double.valueOf(rows);
        }
        resizedWidth = (int) (cols * scale);
        resizedHeight = (int) (rows * scale);

        BufferedImage resizedImage = new BufferedImage(resizedWidth, resizedHeight, BufferedImage.TYPE_INT_RGB);
        Graphics graphics = resizedImage.createGraphics();
        graphics.drawImage(image, 0, 0, resizedWidth, resizedHeight, null);

        //绘制到中间
        BufferedImage paddingImage = new BufferedImage(model_input_width, model_input_height, BufferedImage.TYPE_INT_RGB);
        offsetX = (paddingImage.getWidth() - resizedWidth) / 2;
        offsetY = (paddingImage.getWidth() - resizedHeight) / 2;
        Graphics graphics_padding = paddingImage.createGraphics();
        graphics_padding.drawImage(resizedImage, offsetX, offsetY, resizedWidth, resizedHeight, null);

        float[][][] arr = new float[3][model_input_width][model_input_height];
        for (int i = 0; i < model_input_width; i++) {
            for (int j = 0; j < model_input_height; j++) {
                if (i >= offsetY && j >= offsetX && i < offsetY + resizedHeight
                        && j < offsetX + resizedWidth) {
                    int rgb = resizedImage.getRGB(j - offsetX, i - offsetY);
                    Color color = new Color(rgb, true);
                    arr[0][i][j] = color.getRed() / 255.0f;
                    arr[1][i][j] = color.getGreen() / 255.0f;
                    arr[2][i][j] = color.getBlue() / 255.0f;
                } else {
                    arr[0][i][j] = 114.0f / 255;
                    arr[1][i][j] = 114.0f / 255;
                    arr[2][i][j] = 114.0f / 255;
                }
            }
        }

        return arr;
    }


}
