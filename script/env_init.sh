#!/bin/bash
set -e -x
pip install pytest
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install opencv-python==4.6.0.66
pip install opencv-python-headless