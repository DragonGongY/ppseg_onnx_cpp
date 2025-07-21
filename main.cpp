#include <iostream>
#include "ppseg.h"

int main(int, char**){
    PPSeg ppseg("/home/deepseavision/workspace/cpp_projs/ppseg_cpp/pp_mobileseg_base_camvid_1024x1024_model.onnx");
    cv::Mat image = cv::imread("/home/deepseavision/workspace/cpp_projs/ppseg_cpp/test.jpg");
    ppseg.segment(image);
}
