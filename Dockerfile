###############################################################################
FROM ubuntu:20.04
###############################################################################

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -yq build-essential cmake openssl libssl-dev wget unzip libopencv-dev

RUN apt-get install -y gcc-10 g++-10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && true

WORKDIR /home

RUN wget -O poco.zip https://github.com/pocoproject/poco/archive/refs/tags/poco-1.12.4-release.zip && unzip poco.zip -d /tmp && \
    mkdir /tmp/poco-poco-1.12.4-release/cmake-build && cd /tmp/poco-poco-1.12.4-release/cmake-build && cmake .. && \
    cmake --build . -j 6 --config Release && cmake --build . --target install && cd - && rm poco.zip && rm -rf /tmp/poco-poco-1.12.4-release

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | tee /etc/apt/sources.list.d/intel-openvino-2022.list && \
    apt-get update && apt-get install -yq openvino-2022.3.0 && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

COPY . /img-cls-server/repo/
RUN cd /img-cls-server/repo && mkdir -p /img-cls-server/build && \
    cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -S/img-cls-server/repo -B/img-cls-server/build -G "Unix Makefiles" && \
    cmake --build /img-cls-server/build --config Release --target all -j 8 -- && cp /img-cls-server/build/img-cls-server /img-cls-server && \
    cp -r /img-cls-server/repo/data /img-cls-server/data && rm -rf /img-cls-server/repo/ && rm -rf /img-cls-server/build

WORKDIR /img-cls-server
# CMD ["./img-cls-server"]

CMD ["tail", "-f", "/dev/null"]

LABEL Name=img-cls-server Version=0.0.1