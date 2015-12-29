#include <opencv2/opencv.hpp>
#include <iostream>
#include <tuple>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace py = boost::python;
namespace np = boost::numpy;

cv::Mat convert_to_cvmat(const np::ndarray& img) {
  const long *shape   = img.get_shape();
  const int   height  = shape[0];
  const int   width   = shape[1];
  const int   channel = shape[2];
  unsigned char *data = reinterpret_cast<unsigned char *>(img.get_data());
  cv::Mat img_mat(height, width, CV_8UC(channel));

  img_mat.data = data;
  return img_mat;
}

np::ndarray convert_to_ndarray(std::vector<cv::Mat>imgs) {
  const int   n       = imgs.size();
  const int   h       = imgs.at(0).rows;
  const int   w       = imgs.at(0).cols;
  const int   ch      = imgs.at(0).channels();
  py::tuple   shape   = py::make_tuple(n, h, w, ch);
  np::dtype   dtype   = np::dtype::get_builtin<unsigned char>();
  np::ndarray img     = np::zeros(shape, dtype);
  unsigned char *data = reinterpret_cast<unsigned char *>(img.get_data());

  for (size_t i = 0; i < n; i++) {
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        if (ch == 1) {
          unsigned char pix   = imgs.at(i).at<unsigned char>(x, y);
          const int     index = (i * h * w * ch) + (y * w * ch) + (x * ch) + 0;
          data[index] = pix;
        }
        else if (ch == 3) {
          cv::Vec3b pix = imgs.at(i).at<cv::Vec3b>(x, y);

          for (size_t c = 0; c < ch; c++) {
            const int index = (i * h * w * ch) + (y * w * ch) + (x * ch) + c;
            data[index] = pix[c];
          }
        }
      }
    }
  }

  return img;
}

py::tuple divide_to_patches(
  const int       & stride,
  const int       & sat_size,
  const int       & map_size,
  const np::ndarray sat_im,
  const np::ndarray map_im) {
  cv::Mat sat_img = convert_to_cvmat(sat_im);
  cv::Mat map_img = convert_to_cvmat(map_im);
  std::vector<cv::Mat> sat_patches, map_patches;

  for (size_t y = 0; y != sat_img.rows - sat_size; y += stride) {
    for (size_t x = 0; x != sat_img.cols - sat_size; x += stride) {
      if (y + sat_size > sat_img.rows) y = sat_img.rows - sat_size;

      if (x + sat_size > sat_img.cols) x = sat_img.cols - sat_size;

      cv::Mat sat_patch = sat_img(cv::Rect(x, y, sat_size, sat_size));

      cv::Mat map_patch = map_img(
        cv::Rect(x + sat_size / 2 - map_size / 2,
                 y + sat_size / 2 - map_size / 2,
                 map_size, map_size));

      int sum_sat_values = 0;

      for (size_t yy = 0; yy < sat_size; yy++) {
        for (size_t xx = 0; xx < sat_size; xx++) {
          cv::Vec3b pix = sat_patch.at<cv::Vec3b>(xx, yy);

          if (pix[0] + pix[1] + pix[2] > 255 * 3) ++sum_sat_values;
        }
      }

      if (sum_sat_values > sat_size * sat_size / 2) continue;

      sat_patches.push_back(sat_patch);
      map_patches.push_back(map_patch);
    }
  }

  return py::make_tuple(convert_to_ndarray(sat_patches),
                        convert_to_ndarray(map_patches));
}

BOOST_PYTHON_MODULE(patches) {
  np::initialize();

  def("divide_to_patches", divide_to_patches);
}
