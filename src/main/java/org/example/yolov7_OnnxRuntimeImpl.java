package org.example;

import ai.onnxruntime.OrtException;

import java.awt.image.BufferedImage;
import java.util.ArrayList;

public interface yolov7_OnnxRuntimeImpl {
    boolean readModel(String onnxPath, String... classes);
    ArrayList<OutResult> Detect(float minConfidence, BufferedImage image) throws OrtException;
    float[][][] img2Float(BufferedImage image);
}
