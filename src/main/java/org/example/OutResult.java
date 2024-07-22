package org.example;

public class OutResult {
    int x1;
    int x2;
    int y1;
    int y2;
    int classId;
    float confidence;

    OutResult(int x1_, int x2_, int y1_, int y2_, int classId_, float confidence_) {
        x1 = x1_;
        x2 = x2_;
        y1 = y1_;
        y2 = y2_;
        classId = classId_;
        confidence = confidence_;
    }
}
