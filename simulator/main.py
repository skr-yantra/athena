#!/usr/bin/env python
import math
import pybullet as p
import time
import pybullet_data

from data import abb_irb120

urdf = abb_irb120()

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 0]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF(urdf, cubeStartPos, cubeStartOrientation, useFixedBase=True)

print(p.getNumJoints(boxId))
for i in range(0, p.getNumJoints(boxId)):
    print("Joint %d: %s" % (i, p.getJointInfo(boxId, i)))

# p.setJointMotorControl2(boxId, 1, controlMode=p.POSITION_CONTROL, targetPosition=math.pi/2)

(l1, l2, l3, l4, l5, l6, f1, f2) = p.calculateInverseKinematics(
    boxId,
    7,
    [0, 0.5, 1],
    p.getQuaternionFromEuler([0, math.pi/2, 0]),
)

p.setJointMotorControlArray(
    boxId,
    [1, 2, 3, 4, 5, 6, 8, 9],
    p.POSITION_CONTROL,
    [l1, l2, l3, l4, l5, l6, f1, f2]
)

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
