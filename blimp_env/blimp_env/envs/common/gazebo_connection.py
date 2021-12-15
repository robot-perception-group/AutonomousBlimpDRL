""" handle gazebo ros connection """
#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import (
    DeleteModel,
    SetPhysicsProperties,
    SetPhysicsPropertiesRequest,
    SpawnModel,
)
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
from std_srvs.srv import Empty


class GazeboConnection:
    """handel connection to gazebo simulator"""

    def __init__(self, start_init_physics_parameters=True, reset_world_or_sim="WORLD"):
        rospy.loginfo("GazeboConnection Initializing ...")

        self._max_update_rate = Float64(400.0)
        self._ode_config = ODEPhysics()
        self._time_step = Float64(0.005)  # 0.001

        self.delete = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_simulation_proxy = rospy.ServiceProxy(
            "/gazebo/reset_simulation", Empty
        )
        self.reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        # Setup the Gravity Control system
        service_name = "/gazebo/set_physics_properties"
        rospy.logdebug("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        rospy.logdebug("Service Found " + str(service_name))

        self.set_physics = rospy.ServiceProxy(service_name, SetPhysicsProperties)
        self.start_init_physics_parameters = start_init_physics_parameters
        self.reset_world_or_sim = reset_world_or_sim
        self.init_values()

        # We always pause the simulation, important for legged robots learning
        self.pause_sim()
        rospy.loginfo("GazeboConnection Initialize Finished")

    def spawn_model(self):
        """spawn urdf model with ros service call"""
        rospy.logdebug("SPAWNMODEL START")
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        try:
            self.spawn()
        except rospy.ServiceException as err:
            print("/gazebo/spawn_urdf_model service call failed", err)

        rospy.logdebug("SPAWNMODEL FINISH")

    def delete_model(self):
        """delete urdf model with ros service call"""
        rospy.logdebug("DELETEMODEL START")
        rospy.wait_for_service("/gazebo/delete_model")
        try:
            self.delete()
        except rospy.ServiceException as err:
            print("/gazebo/delete_model service call failed", err)

        rospy.logdebug("DELETEMODEL FINISH")

    def pause_sim(self):
        """pause simulation with ros service call"""
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as err:
            print("/gazebo/pause_physics service call failed", err)

        rospy.logdebug("PAUSING FINISH")

    def unpause_sim(self):
        """unpause simulation with ros service call"""
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as err:
            print("/gazebo/unpause_physics service call failed", err)

        rospy.logdebug("UNPAUSING FiNISH")

    def reset_sim(self):
        """
        This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
        """
        if self.reset_world_or_sim == "SIMULATION":
            rospy.logerr("SIMULATION RESET")
            self.reset_simulation()
        elif self.reset_world_or_sim == "WORLD":
            rospy.logerr("WORLD RESET")
            self.reset_world()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            rospy.logerr("NO RESET SIMULATION SELECTED")
        else:
            rospy.logerr("WRONG Reset Option:" + str(self.reset_world_or_sim))

    def reset_simulation(self):
        """reset whole simulation"""
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_simulation service call failed", err)

    def reset_world(self):
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_world service call failed", err)

    def init_values(self):
        """initialize gaz parameters"""

        self.reset_sim()

        if self.start_init_physics_parameters:
            rospy.logdebug("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            rospy.logerr("NOT Initialising Simulation Physics Parameters")

    def init_physics_parameters(self):
        """
        We initialise the physics parameters of the simulation, like gravity,
        friction coeficients and so on.
        """

        self._gravity = Vector3()  # pylint: disable=attribute-defined-outside-init
        self._gravity.x = 0.0
        self._gravity.y = 0.0
        self._gravity.z = -9.81

        self._ode_config.auto_disable_bodies = False
        self._ode_config.sor_pgs_precon_iters = 0
        self._ode_config.sor_pgs_iters = 200  # 50
        self._ode_config.sor_pgs_w = 1.9  # 1.3
        self._ode_config.sor_pgs_rms_error_tol = 0.0
        self._ode_config.contact_surface_layer = 0.001
        self._ode_config.contact_max_correcting_vel = 100  # 0.0
        self._ode_config.cfm = 0.0
        self._ode_config.erp = 0.2
        self._ode_config.max_contacts = 20

        self.update_gravity_call()

    def update_gravity_call(self):
        """update gravity call"""

        self.pause_sim()

        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self._time_step.data
        set_physics_request.max_update_rate = self._max_update_rate.data
        set_physics_request.gravity = self._gravity
        set_physics_request.ode_config = self._ode_config

        rospy.logdebug(str(set_physics_request.gravity))

        result = self.set_physics(set_physics_request)
        rospy.logdebug(
            "Gravity Update Result=="
            + str(result.success)
            + ",message=="
            + str(result.status_message)
        )

        self.unpause_sim()

    def change_gravity(self, gravity):
        """change gravity"""
        self._gravity.x, self._gravity.y, self._gravity.z = gravity

        self.update_gravity_call()
