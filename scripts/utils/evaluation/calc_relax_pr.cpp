#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace p  = boost::python;
namespace np = boost::numpy;

int relax_precision(const np::ndarray& predict,
                    const np::ndarray& label, const int& relax) {
  const int h_lim             = predict.shape(1);
  const int w_lim             = predict.shape(0);
  const int32_t *predict_data =
    reinterpret_cast<int32_t *>(predict.get_data());
  const int32_t *label_data =
    reinterpret_cast<int32_t *>(label.get_data());

  int true_positive = 0;

  for (int y = 0; y < h_lim; ++y) {
    for (int x = 0; x < w_lim; ++x) {
      const int32_t pred_val = predict_data[y * w_lim + x];

      if (pred_val == 1) {
        const int st_y = y - relax >= 0 ? y - relax : 0;
        const int en_y = y + relax < h_lim ? y + relax : h_lim - 1;
        const int st_x = x - relax >= 0 ? x - relax : 0;
        const int en_x = x + relax < w_lim ? x + relax : w_lim - 1;
        int sum        = 0;

        for (int yy = st_y; yy <= en_y; ++yy) {
          for (int xx = st_x; xx <= en_x; ++xx) {
            if (label_data[yy * w_lim + xx] == 1) {
              ++sum;
              ++true_positive;
              break;
            }
          }

          if (sum > 0) break;
        }
      }
    }
  }

  return true_positive;
}

int relax_recall(const np::ndarray predict,
                 const np::ndarray label, const int& relax) {
  const int h_lim             = label.shape(1);
  const int w_lim             = label.shape(0);
  const int32_t *predict_data =
    reinterpret_cast<int32_t *>(predict.get_data());
  const int32_t *label_data =
    reinterpret_cast<int32_t *>(label.get_data());

  int true_positive = 0;

  for (int y = 0; y < h_lim; ++y) {
    for (int x = 0; x < w_lim; ++x) {
      const int32_t label_val = label_data[y * w_lim + x];

      if (label_val > 0) {
        const int st_y = y - relax >= 0 ? y - relax : 0;
        const int en_y = y + relax < h_lim ? y + relax : h_lim - 1;
        const int st_x = x - relax >= 0 ? x - relax : 0;
        const int en_x = x + relax < w_lim ? x + relax : w_lim - 1;
        int sum        = 0;

        for (int yy = st_y; yy <= en_y; ++yy) {
          for (int xx = st_x; xx <= en_x; ++xx) {
            if (predict_data[yy * w_lim + xx] == 1) {
              ++sum;
              ++true_positive;
              break;
            }
          }

          if (sum > 0) break;
        }
      }
    }
  }

  return true_positive;
}

BOOST_PYTHON_MODULE(evaluation) {
  np::initialize();

  def("relax_precision", relax_precision);
  def("relax_recall",    relax_recall);
}
