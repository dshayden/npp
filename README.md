NPP
--------

Python code for Nonparametric Parts (NPP) Model from CVPR 2020:

Hayden, David S., Jason Pacheco, and John W. Fisher. "Nonparametric Object and
Parts Modeling With Lie Group Dynamics." Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition. 2020.

Getting Started
--------
Example Nonparametric Parts inference can be run by doing the following:

    git pull git@github.com:dshayden/npp.git

    cd npp

    pip install -r requirements.txt

    pip install -e .

    sh example.sh

This will run initialization and inference on the se2_randomwalk3 dataset.
Results for the final sample, visualized as PNG files, will be placed in the
directory example_inference/visualization. Samples are placed in the directory
example_inference/samples, and can be loaded for analyis using
npp.SED.loadSample. See script/drawSample.py for code on how to do this.

Coming Soon
--------
Datasets (2D and 3D) used in the paper.
