FROM nucci2/lmod:8.4.3

SHELL ["/usr/bin/bash", "-c"]

LABEL author="Jason C. Nucciarone" \
    maintainer="Jason C. Nucciarone" \
    version="v0.1"

# Install Python 3.8.8 in the home directory
RUN yum update -y && yum install findutils -y && \
    cd $HOME && \
    wget https://www.python.org/ftp/python/3.8.8/Python-3.8.8.tar.xz -O - | tar -xJv && \
    cd Python-3.8.8 && \
    ./configure --enable-shared --enable-optimizations --prefix=$HOME/python-3.8.8 && \
    make && make install && \
    cd $HOME && \
    rm -rf Python-3.8.8 Python-3.8.8.tar.xz && \
    mkdir -p /etc/modulefiles/python && \
    ln -s $HOME/python-3.8.8/bin/python3 $HOME/python-3.8.8/bin/python

# Copy module file into Docker container
COPY 3.8.8.lua /etc/modulefiles/python/3.8.8.lua

# Add /etc/modulefiles to MODULEPATH
ENV MODULEPATH=/etc/modulefiles:$MODULEPATH

# Start bash when using `docker run`
CMD ["/usr/bin/bash"]
