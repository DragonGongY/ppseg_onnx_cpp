#include "ppseg.h"

PPSeg::PPSeg(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PPSeg"),
      session_options_(),
      session_(env_, model_path.c_str(), session_options_) {}

void PPSeg::segment(const cv::Mat& image) {
  cv::Mat input_tensor = preprocess(image);  // [1,C,H,W]
  if (input_tensor.empty()) {
    std::cerr << "Failed to preprocess image" << std::endl;
    return;
  }

  std::vector<int64_t> input_dims = {1, 3, input_size_.height,
                                     input_size_.width};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value onnx_input = Ort::Value::CreateTensor<float>(
      memory_info, (float*)input_tensor.data,
      input_tensor.total() * input_tensor.elemSize(), input_dims.data(),
      input_dims.size());

  // 推理
  auto output_tensors =
      session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &onnx_input,
                   1, output_names_.data(), output_names_.size());

  // 获取输出并 argmax
  float* output_data = output_tensors[0].GetTensorMutableData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int out_c = output_shape[1];
  int out_h = output_shape[2];
  int out_w = output_shape[3];

  std::vector<uint8_t> pred(out_h * out_w);
  for (int h = 0; h < out_h; ++h) {
    for (int w = 0; w < out_w; ++w) {
      int max_idx = 0;
      float max_val = output_data[0 * out_h * out_w + h * out_w + w];
      for (int c = 1; c < out_c; ++c) {
        float val = output_data[c * out_h * out_w + h * out_w + w];
        if (val > max_val) {
          max_val = val;
          max_idx = c;
        }
      }
      pred[h * out_w + w] = static_cast<uint8_t>(max_idx);
    }
  }

  cv::Mat mask(out_h, out_w, CV_8UC1, pred.data());
  std::vector<uint8_t> palette = getColorMapList(256);

  cv::Mat color_mask(out_h, out_w, CV_8UC3);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      int cls = mask.at<uint8_t>(y, x);
      color_mask.at<cv::Vec3b>(y, x) = {
          palette[cls * 3 + 2], palette[cls * 3 + 1], palette[cls * 3 + 0]};
    }
  }

  std::string save_path = "output.png";
  cv::imwrite(save_path, color_mask);
  std::cout << "Saved prediction: " << save_path << std::endl;

  // === 在原图上画出mask轮廓 ===
  // 1. 先将mask resize回原图大小
  cv::Mat mask_resized;
  cv::resize(mask, mask_resized, image.size(), 0, 0, cv::INTER_NEAREST);

  // 2. 查找轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask_resized, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // 3. 在原图上画轮廓
  cv::Mat image_with_contour = image.clone();
  cv::drawContours(image_with_contour, contours, -1, cv::Scalar(0, 0, 255),
                   2);  // 红色

  // 4. 保存结果
  std::string contour_path = "output_contour.png";
  cv::imwrite(contour_path, image_with_contour);
  std::cout << "Saved contour image: " << contour_path << std::endl;
}

std::vector<uint8_t> PPSeg::getColorMapList(
    int num_classes, const std::vector<uint8_t>& custom_color) {
  num_classes += 1;
  std::vector<uint8_t> color_map(num_classes * 3, 0);

  for (int i = 0; i < num_classes; ++i) {
    int lab = i;
    int j = 0;
    while (lab) {
      color_map[i * 3 + 0] |= ((lab >> 0) & 1) << (7 - j);
      color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j);
      color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j);
      ++j;
      lab >>= 3;
    }
  }

  color_map.erase(color_map.begin(), color_map.begin() + 3);  // 去掉背景
  if (!custom_color.empty()) {
    for (size_t i = 0; i < custom_color.size() && i < color_map.size(); ++i) {
      color_map[i] = custom_color[i];
    }
  }
  return color_map;
}

cv::Mat PPSeg::preprocess(const cv::Mat& image) {
  if (image.empty()) {
    std::cerr << "Failed to read image: " << image << std::endl;
    return {};
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::Mat resized_image;
  cv::resize(image, resized_image, input_size_);

  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  cv::split(resized_image, channels);

  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - mean_[c]) / std_[c];
  }

  cv::Mat chw;
  cv::vconcat(channels, chw);

  cv::Mat result =
      chw.reshape(1, {1, 3, input_size_.height, input_size_.width});
  return result;
}
