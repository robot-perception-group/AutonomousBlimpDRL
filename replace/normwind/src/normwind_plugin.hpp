/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 * Copyright 2016 Geoffrey Hunter <gbmhunter@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef ROTORS_GAZEBO_PLUGINS_NORMWIND_PLUGIN_H
#define ROTORS_GAZEBO_PLUGINS_NORMWIND_PLUGIN_H

#include <string>
#include <random>
#include <gazebo/common/common.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

#include <mav_msgs/default_topics.h>  // This comes from the mav_comm repo
#include <geometry_msgs/Point.h>
#include "common.h"

#include "WindSpeed.pb.h"             // Wind speed message
#include "WrenchStamped.pb.h"         // Wind force message

namespace gazebo {
typedef const boost::shared_ptr<const gz_mav_msgs::WindSpeed>
      GzWindSpeedMsgPtr;

// Default values
static const std::string kDefaultFrameId = "world";
static const std::string kDefaultLinkName = "base_link";
static const std::string kDefaultWindSpeedPubTopic = "wind_speed";
static const std::string kDefaultWindStateTopic = "wind_state";

static constexpr double kDefaultWindSpeedMean = 0.0;
static constexpr int kDefaultTurbulenceLevel = 0;

static const ignition::math::Vector3d kDefaultWindDirectionMean = ignition::math::Vector3d(1, 0, 0);
static const ignition::math::Vector3d kDefaultWindGustDirectionMean = ignition::math::Vector3d(0, 1, 0);
static constexpr double kDefaultWindDirectionVariance = 0.0;
static constexpr double kDefaultWindGustDirectionVariance = 0.0;

static constexpr bool kDefaultUseCustomStaticWindField = false;



/// \brief    This gazebo plugin simulates wind acting on a model.
/// \details  This plugin publishes on a Gazebo topic and instructs the ROS interface plugin to
///           forward the message onto ROS.
class GazeboWindPlugin : public ModelPlugin {
 public:
  GazeboWindPlugin()
      : ModelPlugin(),
        namespace_(kDefaultNamespace),
        wind_force_pub_topic_(mav_msgs::default_topics::EXTERNAL_FORCE),
        wind_speed_pub_topic_(mav_msgs::default_topics::WIND_SPEED),
        wind_speed_mean_(kDefaultWindSpeedMean),
        wind_turbulence_level(kDefaultTurbulenceLevel),
        wind_direction_mean_(kDefaultWindDirectionMean),
        use_custom_static_wind_field_(kDefaultUseCustomStaticWindField),
        frame_id_(kDefaultFrameId),
        link_name_(kDefaultLinkName),
        node_handle_(nullptr),
        pubs_and_subs_created_(false) {}

  virtual ~GazeboWindPlugin();

 protected:

  /// \brief Load the plugin.
  /// \param[in] _model Pointer to the model that loaded this plugin.
  /// \param[in] _sdf SDF element that describes the plugin.
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

  /// \brief Called when the world is updated.
  /// \param[in] _info Update timing information.
  void OnUpdate(const common::UpdateInfo& /*_info*/);

  /// \brief    update wind state.
  /// \details  update wind state from subscriber
  void WindStateCallback(GzWindSpeedMsgPtr& wind_state_msg);
  std::string wind_state_sub_topic_;
  gazebo::transport::SubscriberPtr wind_state_sub_;

 private:

  // a function for the turbulence value at altitude
  double POEValue(int s, double h);


  /// \brief    Flag that is set to true once CreatePubsAndSubs() is called, used
  ///           to prevent CreatePubsAndSubs() from be called on every OnUpdate().
  bool pubs_and_subs_created_;

  /// \brief    Creates all required publishers and subscribers, incl. routing of messages to/from ROS if required.
  /// \details  Call this once the first time OnUpdate() is called (can't
  ///           be called from Load() because there is no guarantee GazeboRosInterfacePlugin has
  ///           has loaded and listening to ConnectGazeboToRosTopic and ConnectRosToGazeboTopic messages).
  void CreatePubsAndSubs();

  /// \brief    Pointer to the update event connection.
  event::ConnectionPtr update_connection_;

  physics::WorldPtr world_;
  physics::ModelPtr model_;
  physics::LinkPtr link_;

  std::string namespace_;

  std::string frame_id_;
  std::string link_name_;
  std::string wind_force_pub_topic_;
  std::string wind_speed_pub_topic_;

  common::Time previousRun;

  double wind_speed_mean_;
  int wind_turbulence_level;

  //ignition::math::Vector3d xyz_offset_;
  ignition::math::Vector3d wind_direction_mean_;
  //ignition::math::Vector3d wind_gust_direction_mean_;
  //double wind_direction_variance_;
  //double wind_gust_direction_variance_;
  std::normal_distribution<double> GaussianRandomNumber;
  std::default_random_engine randomGen;

  //std::default_random_engine wind_direction_generator_;
  //std::normal_distribution<double> wind_direction_distribution_X_;
  //std::normal_distribution<double> wind_direction_distribution_Y_;
  //std::normal_distribution<double> wind_direction_distribution_Z_;
  //std::default_random_engine wind_gust_direction_generator_;
  //std::normal_distribution<double> wind_gust_direction_distribution_X_;
  //std::normal_distribution<double> wind_gust_direction_distribution_Y_;
  //std::normal_distribution<double> wind_gust_direction_distribution_Z_;

  //common::Time wind_gust_end_;
  //common::Time wind_gust_start_;

  /// \brief    Variables for custom wind field generation.
  bool use_custom_static_wind_field_;
  float min_x_;
  float min_y_;
  int n_x_;
  int n_y_;
  float res_x_;
  float res_y_;
  std::vector<float> vertical_spacing_factors_;
  std::vector<float> bottom_z_;
  std::vector<float> top_z_;
  std::vector<float> u_;
  std::vector<float> v_;
  std::vector<float> w_;
  
  /// \brief  Reads wind data from a text file and saves it.
  /// \param[in] custom_wind_field_path Path to the wind field from ~/.ros.
  void ReadCustomWindField(std::string& custom_wind_field_path);
  
  /// \brief  Functions for trilinear interpolation of wind field at aircraft position.
  
  /// \brief  Linear interpolation
  /// \param[in]  position y-coordinate of the target point.
  ///             values Pointer to an array of size 2 containing the wind values
  ///                    of the two points to interpolate from (12 and 13).
  ///             points Pointer to an array of size 2 containing the y-coordinate 
  ///                    of the two points to interpolate from.
  ignition::math::Vector3d LinearInterpolation(double position, ignition::math::Vector3d * values, double* points) const;
  
  /// \brief  Bilinear interpolation
  /// \param[in]  position Pointer to an array of size 2 containing the x- and 
  ///                      y-coordinates of the target point.
  ///             values Pointer to an array of size 4 containing the wind values 
  ///                    of the four points to interpolate from (8, 9, 10 and 11).
  ///             points Pointer to an array of size 14 containing the z-coordinate
  ///                    of the eight points to interpolate from, the x-coordinate 
  ///                    of the four intermediate points (8, 9, 10 and 11), and the 
  ///                    y-coordinate of the last two intermediate points (12 and 13).
  ignition::math::Vector3d BilinearInterpolation(double* position, ignition::math::Vector3d * values, double* points) const;
  
  /// \brief  Trilinear interpolation
  /// \param[in]  link_position Vector3 containing the x, y and z-coordinates
  ///                           of the target point.
  ///             values Pointer to an array of size 8 containing the wind values of the 
  ///                    eight points to interpolate from (0, 1, 2, 3, 4, 5, 6 and 7).
  ///             points Pointer to an array of size 14 containing the z-coordinate          
  ///                    of the eight points to interpolate from, the x-coordinate 
  ///                    of the four intermediate points (8, 9, 10 and 11), and the 
  ///                    y-coordinate of the last two intermediate points (12 and 13).
  ignition::math::Vector3d TrilinearInterpolation(ignition::math::Vector3d link_position, ignition::math::Vector3d * values, double* points) const;
  
  gazebo::transport::PublisherPtr wind_force_pub_;
  gazebo::transport::PublisherPtr wind_speed_pub_;

  gazebo::transport::NodePtr node_handle_;

  /// \brief    Gazebo message for sending wind data.
  /// \details  This is defined at the class scope so that it is re-created
  ///           everytime a wind message needs to be sent, increasing performance.
  gz_geometry_msgs::WrenchStamped wrench_stamped_msg_;

  /// \brief    Gazebo message for sending wind speed data.
  /// \details  This is defined at the class scope so that it is re-created
  ///           everytime a wind speed message needs to be sent, increasing performance.
  gz_mav_msgs::WindSpeed wind_speed_msg_;

  const double POE[8][12]={
        {  500.0, 1750.0, 3750.0, 7500.0, 15000.0, 25000.0, 35000.0, 45000.0, 55000.0, 65000.0, 75000.0, 80000.0},
        {    3.2, 2.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {    4.2, 3.6, 3.3, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {    6.6, 6.9, 7.4, 6.7, 4.6, 2.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0},
        {    8.6, 9.6, 10.6, 10.1, 8.0, 6.6, 5.0, 4.2, 2.7, 0.0, 0.0, 0.0},
        {   11.8, 13.0, 16.0, 15.1, 11.6, 9.7, 8.1, 8.2, 7.9, 4.9, 3.2, 2.1},
        {   15.6, 17.6, 23.0, 23.6, 22.1, 20.0, 16.0, 15.1, 12.1, 7.9, 6.2, 5.1},
        {   18.7, 21.5, 28.4, 30.2, 30.7, 31.0, 25.2, 23.1, 17.5, 10.7, 8.4, 7.2}};
};
}

#endif // ROTORS_GAZEBO_PLUGINS_GAZEBO_WIND_PLUGIN_H
