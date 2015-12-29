#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace py = boost::python;
namespace np = boost::numpy;

cv::Mat transform(
  const cv::Mat& img,
  const bool   & fliplr,
  const bool   & rotate,
  const int    & angle,
  const bool   & norm,
  const int    & out_h,
  const int    & out_w,
  const int    & channels) {
  cv::Mat img_mat(img.cols, img.rows, CV_64FC(channels));

  img.convertTo(img_mat, CV_64F);

  // Flipping L-R
  if (fliplr) cv::flip(img_mat, img_mat, 1);

  // Rotation with random arngle
  if (rotate) {
    cv::Point2f center(img_mat.cols / 2., img_mat.rows / 2.);
    cv::Mat     r = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img_mat, img_mat, r, cv::Size(img_mat.cols, img_mat.rows),
                   cv::INTER_NEAREST);
  }

  // Cropping
  cv::Rect roi(img_mat.cols / 2 - out_w / 2, img_mat.rows / 2 - out_h / 2,
               out_w, out_h);
  cv::Mat crop(out_w, out_h, CV_64FC(channels));
  img_mat(roi).copyTo(crop);

  // patch-wise mean subtraction
  if (norm && (img.channels() == 3)) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(crop, mean, stddev);

    cv::Mat *slice = new cv::Mat[channels];
    cv::split(crop, slice);

    for (int c = 0; c < channels; ++c) {
      cv::subtract(slice[c], mean[c], slice[c]);
      slice[c] /= stddev[c] + 1E-5;
    }
    cv::merge(slice, channels, crop);
    delete[] slice;
  }

  return crop;
}

cv::Mat extract_cvmat(const np::ndarray& img, const int& n) {
  size_t rows         = img.shape(1);
  size_t cols         = img.shape(2);
  size_t channels     = img.shape(3);
  unsigned char *data = reinterpret_cast<unsigned char *>(img.get_data());

  data += n * rows * cols * channels;
  cv::Mat img_mat(rows, cols, CV_8UC(channels));
  unsigned char *out_data = reinterpret_cast<unsigned char *>(img_mat.data);
  std::memcpy(out_data, data, rows * cols * channels);

  return img_mat;
}

void set_values(
  const cv::Mat& mat,
  np::ndarray    img,
  const int    & n) {
  size_t rows     = mat.rows;
  size_t cols     = mat.cols;
  size_t channels = mat.channels();

  assert(rows == img.shape(1));
  assert(cols == img.shape(2));
  assert(channels == img.shape(3));

  double *mat_data = reinterpret_cast<double *>(mat.data);
  float  *img_data = reinterpret_cast<float *>(img.get_data());

  for (size_t y = 0; y < rows; ++y) {
    for (size_t x = 0; x < cols; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        int bind = n * rows * cols * channels + y * cols * channels + x *
                   channels + c;
        int ind = y * cols * channels + x * channels + c;
        img_data[bind] = float(mat_data[ind]);
      }
    }
  }
}

void batch_transform(
  const np::ndarray& sat,
  const np::ndarray& map,
  np::ndarray      & sat_out,
  np::ndarray      & map_out,
  const bool       & fliplr,
  const bool       & rotate,
  const bool       & norm,
  const int        & sat_out_h,
  const int        & sat_out_w,
  const int        & sat_channels,
  const int        & map_out_h,
  const int        & map_out_w) {
  size_t num = sat.shape(0);

  for (size_t n = 0; n < num; ++n) {
    cv::Mat sat_img  = extract_cvmat(sat, n);
    cv::Mat map_img  = extract_cvmat(map, n);
    int     angle    = rand() % 360;
    cv::Mat sat_crop = transform(sat_img,
                                 fliplr,
                                 rotate,
                                 angle,
                                 norm,
                                 sat_out_h,
                                 sat_out_w,
                                 sat_channels);
    cv::Mat map_crop = transform(map_img,
                                 fliplr,
                                 rotate,
                                 angle,
                                 norm,
                                 map_out_h,
                                 map_out_w,
                                 1);
    set_values(sat_crop, sat_out, n);
    set_values(map_crop, map_out, n);
  }
}

BOOST_PYTHON_MODULE(transform) {
  np::initialize();
  py::def("batch_transform", batch_transform);
}
