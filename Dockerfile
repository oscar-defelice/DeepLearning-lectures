# Let's use the image tensorflow-notebook to build our image on top of it
FROM jupyter/tensorflow-notebook:python-3.9.13

LABEL Oscar de Felice <oscardefelice@gmail.com>

USER root
RUN apt-get update \
    && apt-get install -y build-essential libssl-dev cmake python3-h5py --no-install-recommends

# Install pip
RUN pip install --upgrade pip

COPY requirements.txt .
# Install requirements
RUN pip install -U --no-cache-dir -r requirements.txt

# Let's change to  "$NB_USER" command so the image runs as a non root user by default
USER $NB_UID

# Let's define this parameter to install jupyter lab instead of the default 
# juyter notebook command so we don't have to use it when running the container
# with the option -e
ENV JUPYTER_ENABLE_LAB=yes