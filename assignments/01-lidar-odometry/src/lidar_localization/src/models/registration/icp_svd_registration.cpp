/*
 * @Description: 基于SVD方式实现的ICP
 * @Author: Zhou Jimiao
 * @Date: 2021-01-27 09:48:40
 */


#include "lidar_localization/models/registration/icp_svd_registration.hpp"

#include <vector>
#include <pcl/kdtree/kdtree_flann.h>

#include "glog/logging.h"

namespace lidar_localization {

    ICPSVD_Registration::ICPSVD_Registration(const YAML::Node &node) :
        input_target_(new CloudData::CLOUD()) {
        float max_corr_dist = node["max_corr_dist"].as<float>();
        float trans_eps = node["trans_eps"].as<float>();
        float euc_fitness_eps = node["euc_fitness_eps"].as<float>();
        int max_iter = node["max_iter"].as<int>();

        SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
    }
    ICPSVD_Registration::ICPSVD_Registration(float max_corr_dist,
                                             float trans_eps,
                                             float euc_fitness_eps,
                                             int max_iter) :
        input_target_(new CloudData::CLOUD()) {
        SetRegistrationParam(max_corr_dist, trans_eps, euc_fitness_eps, max_iter);
    }

    bool ICPSVD_Registration::SetInputTarget(const lidar_localization::CloudData::CLOUD_PTR &input_target) {
        input_target_ = input_target;
        LOG(INFO) << input_target_->size() << std::endl;
    }

    bool ICPSVD_Registration::ScanMatch(const lidar_localization::CloudData::CLOUD_PTR &input_source,
                                        const Eigen::Matrix4f &predict_pose,
                                        lidar_localization::CloudData::CLOUD_PTR &result_cloud_ptr,
                                        Eigen::Matrix4f &result_pose) {
        int max_matched_points = (input_source->points.size() > 200) ? 200 : input_source->points.size();
//        int max_matched_points = input_source->points.size();
        int select_step = input_source->points.size() / max_matched_points;

        double factor = 9.0;

        pcl::KdTreeFLANN<CloudData::POINT> kd_tree_flann;
        kd_tree_flann.setInputCloud(input_target_);

        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pts_in_source;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pts_in_target;
        pts_in_source.reserve(max_matched_points);
        pts_in_target.reserve(max_matched_points);

        std::cout << predict_pose << std::endl;
        Eigen::Matrix3f predict_R = predict_pose.topLeftCorner<3, 3>();
        Eigen::Matrix<float, 3, 1> predict_t = predict_pose.topRightCorner<3, 1>();

        std::cout << predict_R << "\n" << predict_t << std::endl;

        std::vector<int> nearest_index(1);
        std::vector<float> nearest_dist(1);

        for (int iter = 0; iter < max_iters_; iter++) {
            //1. match points between source and target

            pts_in_source.clear();
            pts_in_target.clear();

            double sum_squared_distance = 0.0;
            double cur_squared_distance = 0.0;
            double last_squared_distance = std::numeric_limits<double>::max();
            double squared_distance_threshold = std::numeric_limits<double>::max();

            std::vector<int> indices;
            std::vector<float> squared_distances;
            for (int i = 0; i < input_source->points.size(); i += select_step) {
                Eigen::Vector3f pt;
                pt << input_source->points[i].x, input_source->points[i].y, input_source->points[i].z;
                pt = predict_R * pt + predict_t;

                if (!kd_tree_flann.nearestKSearch(CloudData::POINT(pt[0], pt[1], pt[2]),
                                                  1, nearest_index, nearest_dist)) {
                    LOG(ERROR) << "No Match Points found..." << std::endl;
                    return false;
                }
                if (nearest_dist[0] < squared_distance_threshold) {
                    indices.push_back(nearest_index[0]);
                    squared_distances.push_back(nearest_dist[0]);

                    pts_in_source.emplace_back(input_source->at(i).x,
                                               input_source->at(i).y,
                                               input_source->at(i).z);
                    pts_in_target.emplace_back(input_target_->at(nearest_index[0]).x,
                                               input_target_->at(nearest_index[0]).y,
                                               input_target_->at(nearest_index[0]).z);

                    sum_squared_distance += nearest_dist[0];
                }
            }

            //2. Solve R && t
            Eigen::Vector3f mu_source(0.0, 0.0, 0.0);
            Eigen::Vector3f mu_target(0.0, 0.0, 0.0);

            for (int i = 0; i < pts_in_source.size(); i++) {
                mu_source += pts_in_source[i];
                mu_target += pts_in_target[i];
            }

            mu_source /= float(pts_in_source.size());
            mu_target /= float(pts_in_target.size());

            //compute W
            Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
            for (int i = 0; i < pts_in_source.size(); i++) {
                W += (pts_in_target[i] - mu_target) * ((pts_in_source[i] - mu_source).transpose());
            }

            //SVD on W
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f U = svd.matrixU();
            Eigen::Matrix3f V = svd.matrixV();

            predict_R = U * (V.transpose());
            if (predict_R.determinant() < 0) {
                predict_R = -predict_R;
            }

            predict_t = mu_target - predict_R * mu_source;

            //3. check if converged
            cur_squared_distance = sum_squared_distance / (double) pts_in_source.size();
//            std::cout << "cur_squared_distance" << cur_squared_distance << std::endl;
            double squared_distance_change = last_squared_distance - cur_squared_distance;

            if (squared_distance_change < trans_eps_ * trans_eps_) {
                break;
            }
            last_squared_distance = cur_squared_distance;
            squared_distance_threshold = factor * cur_squared_distance;
        }

        result_pose << predict_R(0, 0), predict_R(0, 1), predict_R(0, 2), predict_t(0, 0),
            predict_R(1, 0), predict_R(1, 1), predict_R(1, 2), predict_t(1, 0),
            predict_R(2, 0), predict_R(2, 1), predict_R(2, 2), predict_t(2, 0),
            0, 0, 0, 1;
        LOG(INFO) << "\n" << result_pose << std::endl;
        result_cloud_ptr = input_source;
        for(int i = 0; i < input_source->points.size(); i++) {
            Eigen::Vector3f pt(input_source->points[i].x, input_source->points[i].y, input_source->points[i].z);

            pt = predict_R * pt + predict_t;
            result_cloud_ptr->at(i) = pcl::PointXYZ(pt[0], pt[1], pt[2]);
        }

        return true;
    }
    bool ICPSVD_Registration::SetRegistrationParam(float max_corr_dist,
                                                   float trans_eps,
                                                   float euc_fitness_eps,
                                                   int max_iter) {
        max_corr_dist_ = max_corr_dist;
        trans_eps_ = trans_eps;
        euc_fitness_eps_ = euc_fitness_eps;
        max_iters_ = max_iter;

    }
}