ARG CODE_VERSION="3.7-slim"
FROM python:${CODE_VERSION}
LABEL mantainer="Dmitriy Kisil <email: logart1995@gmail.com>"
ADD . /weather_app
WORKDIR /weather_app
# Install native libraries, required for numpy, pandas and scipy

#RUN apk update \
#    && apk add --no-cache --virtual .build-deps  \
#            gfortran \
#            build-base \
#            openblas-dev \
#            bzip2-dev \
#            coreutils \
#            dpkg-dev dpkg \
#            expat-dev \
#            gcc \
#            gdbm-dev \
#            libc-dev \
#            libffi-dev \
#            libnsl-dev \
#            libressl \
#            libressl-dev \
#            libtirpc-dev \
#            linux-headers \
#            make \
#            ncurses-dev \
#            pax-utils \
#            readline-dev \
#            sqlite-dev \
#            tcl-dev \
#            tk \
#            tk-dev \
#            xz-dev \
#            zlib-dev \
#            libxml2-dev \
#            libxslt-dev \
#            musl-dev \
#            libgcc \
#            curl \
#            jpeg-dev \
#            zlib-dev \
#            freetype-dev \
#            lcms2-dev \
#            openjpeg-dev \
#            tiff-dev \
#            tk-dev \
#            tcl-dev \
#            openblas-dev \
#            lapack-dev \
#            musl-dev linux-headers g++ \
#            bash openssh curl ca-certificates openssl less htop \
#    		make wget rsync cython \
#  		    gfortran \
#            build-base libpng-dev freetype-dev libexecinfo-dev openblas-dev libgomp lapack-dev \
#      		libgcc libquadmath musl  \
#        	libgfortran libstdc++ \
#        	py3-scipy py3-scikit-learn \
#
#    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
##    && apk add --upgrade --no-cache \
##        musl-dev linux-headers g++ \
##        bash openssh curl ca-certificates openssl less htop \
##		make wget rsync cython \
##		# libblas-dev liblapack-dev libatlas-base-dev
##		gfortran \
##        build-base libpng-dev freetype-dev libexecinfo-dev openblas-dev libgomp lapack-dev \
##		libgcc libquadmath musl  \
##		libgfortran libstdc++ \
##		lapack-dev \
###		py3-scipy py3-numpy py3-pandas py3-scikit-learn \
#	&&  pip install --no-cache-dir --upgrade pip \
#	&&  pip install numpy==1.18.1 \
##	&&  pip install scipy==1.4.1 \
##	&&  pip install scikit-learn==0.22 \
#	&&  pip install pandas==0.25.3


# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# packages that we need
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "clock.py"]