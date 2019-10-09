#!/usr/bin/env python
import pybullet as pb
from simulator.env.irb120 import IRB120


def _main():
    pb.connect(pb.GUI)
    simulator = IRB120()
    simulator.setup()

    simulator.move_absolute([0.1, 0.1, 1, 0, 0, 0])


if __name__ == '__main__':
    _main()