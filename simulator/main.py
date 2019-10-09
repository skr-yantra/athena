#!/usr/bin/env python
import math

import pybullet as pb
from simulator.env.irb120 import IRB120
import pybullet_data


def _main():
    pb.connect(pb.GUI)
    simulator = IRB120()

    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF("table/table.urdf", [0.6, 0, 0], globalScaling=0.5)
    pb.loadURDF("cube_small.urdf", [0.4, 0, 0.5], globalScaling=0.5)

    simulator.release_gripper()
    simulator.move_absolute([0.445, 0, 0.5, 0, math.pi/2, 0])
    simulator.move_relative([0, 0, -0.14, 0, 0, 0])
    simulator.hold_gripper()
    simulator.move_relative([0, 0, 0.1, 0, 0, 0])
    simulator.move_relative([0.2, 0, 0, 0, 0, 0])
    simulator.move_relative([0, 0, -0.1, 0, 0, 0])
    simulator.release_gripper()
    simulator.move_relative([0, 0, 0.1, 0, 0, 0])
    simulator.move_relative([-0.2, 0, 0, 0, 0, 0])
    simulator.spin()


if __name__ == '__main__':
    _main()