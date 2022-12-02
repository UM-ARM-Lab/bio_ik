#include <bio_ik/bio_ik.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

#include <boost/range/combine.hpp>
#include <optional>

namespace std {
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
  for (auto const &v_i : v) {
    os << v_i << ", ";
  }
  os << "\n";
  return os;
}
}  // namespace std

class PyBioIK {
 public:
  PyBioIK(std::string robot_description)
      : robot_description_(robot_description),
        model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>(robot_description)),
        model_(model_loader_->getModel()) {}

  std::optional<std::vector<double>> ik(std::map<std::string, std::vector<double>> targets, std::string group) {
    // call BioIK
    auto joint_model_group = model_->getJointModelGroup(group);

    bio_ik::BioIKKinematicsQueryOptions opts;
    opts.return_approximate_solution = false;  // optional

    robot_state::RobotState robot_state_ik(model_);
    robot_state_ik.setToDefaultValues();
    robot_state_ik.update();

    moveit::core::GroupStateValidityCallbackFn state_valid_cb =
        [](moveit::core::RobotState *robot_state, const moveit::core::JointModelGroup *joint_group,
           const double *joint_group_variable_values) { return true; };

    for (auto const &[name, p] : targets) {
      tf2::Vector3 position(p[0], p[1], p[2]);
      ROS_INFO_STREAM_NAMED(
          "BIO_IK", "Adding pos goal " << name << ": " << position.x() << "," << position.y() << "," << position.z());
      opts.goals.emplace_back(std::make_unique<bio_ik::PositionGoal>(name, position));
    }
    opts.goals.emplace_back(std::make_unique<bio_ik::MinimalDisplacementGoal>());
    auto const ok =
        robot_state_ik.setFromIK(joint_model_group,              // joints to be used for IK
                                 EigenSTL::vector_Isometry3d(),  // this isn't used, goals are described in opts
                                 std::vector<std::string>(),     // names of the end-effector links
                                 1,                              // take values from YAML
                                 state_valid_cb, opts);
    std::vector<double> out;
    for (auto const &n : joint_model_group->getActiveJointModelNames()) {
      ROS_DEBUG_STREAM_NAMED("BIO_IK", n);
      out.push_back(robot_state_ik.getVariablePosition(n));
    }
    ROS_INFO_STREAM_NAMED("BIO_IK", "ok? " << ok);
    ROS_INFO_STREAM_NAMED("BIO_IK", "q " << out);

    if (!ok) {
      return {};
    }

    return {out};
  }

  robot_model_loader::RobotModelLoaderPtr model_loader_;
  moveit::core::RobotModelConstPtr const model_;
  std::string robot_description_;
};

int main(int argc, char*argv[]) {
  ros::init(argc, argv, "myik");
  PyBioIK bik("hdt_michigan/robot_description");
  bik.ik({{"left_tool", {0.8, 0.3, 0.8}}}, "whole_body");
}