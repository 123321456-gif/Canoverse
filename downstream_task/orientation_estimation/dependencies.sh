cd lib/pointnet2
python setup.py install
cd ../../

cd lib/sphericalmap_utils
python setup.py install
cd ../../


pip install gorilla-core==0.2.6.0
pip install gpustat==1.0.0
pip install --upgrade protobuf
pip install opencv-python-headless scipy  matplotlib open3d trimesh opencv-python natsort pyinstrument pillow scipy numpy==1.26.4
