/*
 * @Description: 基于SVD方式实现的ICP
 * @Author: Zhou Jimiao
 * @Date: 2021-01-27 09:48:25
 */
#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_SVD_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_SVD_REGISTRATION_HPP_

#include <pcl/registration/icp.h>
#include "lidar_localization/models/registration/registration_interface.hpp"

namespace lidar_localization {
class ICPSVD_Registration: public RegistrationInterface {
  public:
    ICPSVD_Registration(const YAML::Node& node);
    ICPSVD_Registration(
      float max_corr_dist, 
      float trans_eps, 
      float euc_fitness_eps, 
      int max_iter
    );

    bool SetInputTarget(const CloudData::CLOUD_PTR& input_target) override;
    bool ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                   const Eigen::Matrix4f& predict_pose, 
                   CloudData::CLOUD_PTR& result_cloud_ptr,
                   Eigen::Matrix4f& result_pose) override;
//protected:

  private:
    bool SetRegistrationParam(
      float max_corr_dist, 
      float trans_eps, 
      float euc_fitness_eps, 
      int max_iter
    );

  private:
//    pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>::Ptr icp_ptr_;
    CloudData::CLOUD_PTR input_target_;
    float max_corr_dist_;
    float trans_eps_;
    float euc_fitness_eps_;
    int max_iters_;

};
}

#endif