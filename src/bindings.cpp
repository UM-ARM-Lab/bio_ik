#include <bio_ik/bio_ik.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pyrosmsg/pyrosmsg.h>

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

  std::optional<std::vector<double>> object_point_ik(std::string tool_name, Eigen::Vector3d goal,
                                                     Eigen::Vector3d object_point_offset, std::string group) {
    std::map<std::string, Eigen::Vector3d> targets{{"object_point", goal}};
    auto opts = make_opts(targets);

    auto jmg = model_->getJointModelGroup(group);

    robot_state::RobotState state(model_);
    state.setToDefaultValues();
    state.update();

    Eigen::Isometry3d shape_pose{Eigen::Isometry3d::Identity()};
    shape_pose.translate(object_point_offset);

    auto shape = std::make_shared<shapes::Sphere>(0.005);

    state.attachBody("object_point", Eigen::Isometry3d::Identity(), {shape}, {shape_pose}, std::vector<std::string>{},
                     tool_name);

    return ik_from_state(jmg, state, opts);
  }

  std::optional<std::vector<double>> ik_from(std::map<std::string, Eigen::Vector3d> targets, std::vector<double> start,
                                             std::string group) {
    // call BioIK
    auto opts = make_opts(targets);

    auto jmg = model_->getJointModelGroup(group);

    robot_state::RobotState state(model_);
    state.setVariablePositions(jmg->getActiveJointModelNames(), start);
    state.update();

    return ik_from_state(jmg, state, opts);
  }

  std::optional<std::vector<double>> ik(std::map<std::string, Eigen::Vector3d> targets, std::string group) {
    // call BioIK
    auto opts = make_opts(targets);

    auto jmg = model_->getJointModelGroup(group);

    robot_state::RobotState state(model_);
    state.setToDefaultValues();
    state.update();

    return ik_from_state(jmg, state, opts);
  }

  std::unique_ptr<bio_ik::BioIKKinematicsQueryOptions> make_opts(
      std::map<std::string, Eigen::Vector3d> &targets) const {
    auto opts = std::make_unique<bio_ik::BioIKKinematicsQueryOptions>();
    opts->replace = true;                       // needed to replace the default goals!!!
    opts->return_approximate_solution = false;  // optional
    for (auto const &[name, p] : targets) {
      tf2::Vector3 position(p(0), p(1), p(2));
      ROS_DEBUG_STREAM_NAMED(
          "BIO_IK", "Adding pos goal " << name << ": " << position.x() << "," << position.y() << "," << position.z());
      opts->goals.emplace_back(std::make_unique<bio_ik::PositionGoal>(name, position));
    }
    opts->goals.emplace_back(std::make_unique<bio_ik::MinimalDisplacementGoal>());
    return opts;
  }

  std::optional<std::vector<double>> ik_from_state(robot_state::JointModelGroup const *jmg,
                                                   robot_state::RobotState &state,
                                                   std::unique_ptr<bio_ik::BioIKKinematicsQueryOptions> const &opts) {
    moveit::core::GroupStateValidityCallbackFn state_valid_cb =
        [](moveit::core::RobotState *robot_state, const moveit::core::JointModelGroup *joint_group,
           const double *joint_group_variable_values) { return true; };

    auto const ok = state.setFromIK(jmg,                            // joints to be used for IK
                                    EigenSTL::vector_Isometry3d(),  // this isn't used, goals are described in opts
                                    std::vector<std::string>(),     // names of the end-effector links
                                    0,                              // take values from YAML
                                    state_valid_cb, *opts);
    std::vector<double> out;
    for (auto const &n : jmg->getActiveJointModelNames()) {
      ROS_DEBUG_STREAM_NAMED("BIO_IK", n);
      out.push_back(state.getVariablePosition(n));
    }
    ROS_DEBUG_STREAM_NAMED("BIO_IK", "ok? " << ok);
    ROS_DEBUG_STREAM_NAMED("BIO_IK", "q " << out);

    if (!ok) {
      return {};
    }

    return {out};
  }

  robot_model_loader::RobotModelLoaderPtr model_loader_;
  moveit::core::RobotModelConstPtr const model_;
  std::string robot_description_;
};

namespace py = pybind11;

PYBIND11_MODULE(pybio_ik, m) {
  py::class_<PyBioIK>(m, "BioIK")
      .def(py::init<std::string>(), py::arg("robot_description"))
      .def("ik", &PyBioIK::ik, py::arg("targets"), py::arg("group_name"))
      .def("ik_from", &PyBioIK::ik_from, py::arg("targets"), py::arg("start"), py::arg("group_name"))
      .def("object_point_ik", &PyBioIK::object_point_ik, py::arg("link_name"), py::arg("goal"), py::arg("offset"),
           py::arg("group_name"))
      //
      ;
}
