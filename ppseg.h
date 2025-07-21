#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class PPSeg {
 public:
  PPSeg(const std::string& model_path);

  cv::Mat segment(const cv::Mat& image);

  std::vector<uint8_t> getColorMapList(
      int num_classes, const std::vector<uint8_t>& custom_color = {});

  cv::Mat preprocess(const cv::Mat& image);

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;

  std::vector<const char*> input_names_ = {"x"};
  std::vector<const char*> output_names_ = {"save_infer_model/scale_0.tmp_0"};
  std::vector<float> mean_ = {0.5, 0.5, 0.5};
  std::vector<float> std_ = {0.5, 0.5, 0.5};
  cv::Size input_size_ = {1024, 1024};
};