# Dependencies

## Software 
- CMake
- OpenCV


# OpenCV
## Installiation
Run the following commands to install OpenCV on Ubuntu.
- Cloning both the OpenCV and OpenCV contribution
```
git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
```
- Check the version of OpenCV
``` 
cd opencv && git checkout $cvVersion && cd
```
- Check the version of OpenCV Contribution
```
cd ../ && cd opencv_contrib && git checkout $cvVersion
```
- Create a build directory 
```
cd ../ && cd opencv && mkdir build && cd build
```
- compile OpenCV binaries
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \ 
-D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \ 
-D INSTALL_C_EXAMPLES=ON \  -D INSTALL_PYTHON_EXAMPLES=ON \ 
-D WITH_TBB=ON \  -D WITH_V4L=ON \ 
-D OPENCV_PYTHON3_INSTALL_PATH=$cwd/OpenCV-$cvVersion-py3/lib/python3.5/site-packages \
-D WITH_QT=ON \ 
-D WITH_OPENGL=ON \ 
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \  
-D BUILD_EXAMPLES=ON ..
```
- Make the compiled binaries
```
make -j$(nproc)
```
- Install OpenCV
```
sudo make install
```
- To confirm the installation of OpenCV run
```
python3 -c "import cv2; print(cv2.__version__)"
```

# Build the code
## For Checking
```
cd check
mkdir build && cd build
cmake ..
make
```
## For Image Classification example
```
cd example
mkdir build && cd build
cmake ..
make
```

# Run the executable found in the corresponding build directory
## For Checking
```
./TFLiteCheck
```
## For Image Classification example
```
./TFLiteCheck ../../models/classification/mobilenet_v1_1.0_224_quant.tflite ../../models/classification/labels_mobilenet_quant_v1_224.txt ../../images/classification_example.jpg 
```