# Prepare OpenCV source
mkdir -p ~/builds
cd ~/builds

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip

unzip opencv.zip
unzip opencv_contrib.zip

mv opencv-4.1.0 opencv
mv opencv_contrib-4.1.0 opencv_contrib

# Build OpenCV
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
-D ENABLE_FAST_MATH=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
-D PYTHON_EXECUTABLE=$(which python) \
-D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D BUILD_EXAMPLES=ON ..

make -j$(nproc)
echo $PW | sudo -S make install
echo $PW | sudo -S ldconfig

cd $(dirname $(which python))/../lib/python3.6/site-packages/cv2/python-3.6
echo $PW | sudo -S mv cv2*.so cv2.so

# Return to setup directory
cd $SETUP_DIR
