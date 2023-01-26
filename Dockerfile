# GCC support can be specified at major, minor, or micro version
# (e.g. 8, 8.2 or 8.2.0).
# See https://hub.docker.com/r/library/gcc/ for all supported GCC
# tags from Docker Hub.
# See https://docs.docker.com/samples/library/gcc/ for more on how to use this image
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -yq build-essential cmake openssl libssl-dev wget unzip

RUN apt-get install -y gcc-10 g++-10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && true

# These commands copy your files into the specified directory in the image
# and set that as the working location
# COPY . /usr/src/myapp
WORKDIR /home

RUN wget -O poco.zip https://github.com/pocoproject/poco/archive/refs/tags/poco-1.12.4-release.zip && unzip poco.zip -d /tmp && \
    mkdir /tmp/poco-poco-1.12.4-release/cmake-build && cd /tmp/poco-poco-1.12.4-release/cmake-build && cmake .. && \
    cmake --build . -j 6 --config Release && cmake --build . --target install && cd - && rm poco.zip && rm -rf /tmp/poco-poco-1.12.4-release

# RUN wget -O pytorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip && \
    # unzip pytorch.zip && rm libtorch/build* && cp -r libtorch/* /usr/local && rm -rf libtorch && rm pytorch.zip && ldconfig
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | tee /etc/apt/sources.list.d/intel-openvino-2022.list && \
    apt-get update && apt-get install -yq openvino-2022.3.0 && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

RUN apt-get install -yq libopencv-dev

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# This command compiles your app using GCC, adjust for your source code
# RUN g++ -o myapp main.cpp

# This command runs your application, comment out this line to compile only
# CMD ["./myapp"]
CMD ["tail", "-f", "/dev/null"]

LABEL Name=imgclsserver Version=0.0.1