FROM nvcr.io/nvidia/jax:25.01-py3

# default workdir
WORKDIR /workspace

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

RUN pip install jax-tqdm==0.2.2 jraph==0.0.6.dev0 dotmap==1.3.30 evosax==0.1.5

# Uncomment below if you want jupyter 
# RUN pip install jupyterlab

#for secrets and debug
RUN git config --global --add safe.directory /home/${MYUSER}

CMD ["sleep", "infinity"]