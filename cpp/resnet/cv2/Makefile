CC = g++

CFLAGS = -std=c++11 -O3 -o

OUTPUT = resnet
# OUTPUT = test

PKG_CONFIG = pkg-config

# OPENCV
OPENCV_INCLUDE_DIR = /opt/homebrew/Cellar/opencv/4.11.0_1/include/opencv4
OPENCV_LIBS = `$(PKG_CONFIG) --cflags --libs opencv4`

# compiler
INCLUDE = -I $(OPENCV_INCLUDE_DIR)
OPENCV = $(OPENCV_LIBS)


all: main.cpp resnet.cpp bn.cpp conv.cpp conv_bn.cpp fc.cpp pooling.cpp relu.cpp residual.cpp softmax.cpp
	$(CC) $(CFLAGS) $(OUTPUT) main.cpp resnet.cpp  bn.cpp conv.cpp conv_bn.cpp fc.cpp pooling.cpp relu.cpp residual.cpp softmax.cpp -lcnpy $(INCLUDE) $(OPENCV)  
clean:
	rm -f $(OUTPUT)


# all: main.cpp resnet.cpp bn.cpp conv.cpp conv_bn.cpp fc.cpp pooling.cpp relu.cpp residual.cpp softmax.cpp
# 	$(CC) $(CFLAGS) $(OUTPUT) test.cpp resnet.cpp  bn.cpp conv.cpp conv_bn.cpp fc.cpp pooling.cpp relu.cpp residual.cpp softmax.cpp -lcnpy $(INCLUDE) $(OPENCV)  
# clean:
# 	rm -f $(OUTPUT)