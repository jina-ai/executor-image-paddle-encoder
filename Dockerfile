FROM jinaai/jina:2.0.0rc9

# install git
RUN apt-get -y update && apt-get install -y git && apt-get install -y libgomp1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt && pip uninstall -y pathlib

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]