# Optuna Examples for TSUBAME3.0

This is a tutorial material to use Optuna in the [TSUBAME3.0](https://www.t3.gsic.titech.ac.jp/) infrastructure (unofficial).

This tutorial describes:

- How to launch Optuna storage on an interactive node.
- How to parallelize single node ML training.
- How to parallelize multi-node, MPI-based ML training.

## Launch PostgreSQL in TSUBAME

```console
$ GROUP=<YOUR_GROUP>

$ qrsh -g $GROUP -l s_core=1 -l h_rt=12:00:00
$ module load singularity/2.6.1
$ singularity build postgres.img docker://postgres

$ mkdir postgres_data
$ singularity run -B postgres_data:/var/lib/postgresql/data postgres.img /docker-entrypoint.sh postgres
```

The RDB URL is as follows:
```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>  # e.g., STORAGE_HOST=r7i7n7-cnode00
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
```

## Environment Setup

Build the Horovod image and run a container:

```console
$ module load singularity/2.6.1
$ singularity pull docker://uber/horovod:0.15.2-tf1.12.0-torch1.0.0-py3.5
$ singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

With the container, install Python dependencies under the user directory:

```console
$ pip install --user mpi4py psycopg2-binary

# hvd.broadcast_variables is not supported in the old version of Horovod
$ HOROVOD_WITH_TENSORFLOW=1 pip install --user -U --no-cache-dir horovod
```

To deal with MPI-based learning, you need to install a developing branch of Optuna, because the [MPIStudy](https://github.com/pfnet/optuna/blob/horovod-examples/optuna/integration/mpi.py#L46) class has not been merged to the master.

```console
$ pip uninstall optuna  # If you've already installed Optuna.
$ pip install --user git+https://github.com/pfnet/optuna.git@titech-horovod-examples
```

## Distributed Optimization for Single Node Learning

Let's parallelize a simple Optuna script that optimizes a quadratic function.

Set up the RDB URL and create a study identifier:

```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/

$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

Set up a shell script for qsub command, e.g.:

```console
$ echo "module load singularity/2.6.1" >> run_quadratic.sh
$ echo "singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg" >> run_quadratic.sh
$ echo "python tsubame-optuna-example/quadratic.py $STUDY_NAME $STORAGE_URL" >> run_quadratic.sh
```

You can parallelize the optimization just by submitting multiple jobs.
For example, the following commands simultaneously run three workers in a study.

```console
$ GROUP=<YOUR_GROUP>

$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
$ qsub -g $GROUP -l rt_C.small=1 run_quadratic.sh
```

You can list the history of optimization as follows.
```console
$ python print_study_history.py $STUDY_NAME $STORAGE_URL
```

## Distributed Optimization for MPI-based Learning

Let's parallelize a script written in Horovod and TensorFlow.

Download MNIST data:

```console
$ wget -O ~/mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Here, we'll run the example with interactive node. (You also can consolidate the following commands as a batch job.)

```console
$ GROUP=<YOUR_GROUP>
$ qrsh -g $GROUP -l h_node=1 -l h_rt=01:00:00
```

Run a container:

```console
$ module load singularity/2.6.1
$ singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

Create a study identifier in the container:

```console
$ GROUP=<YOUR_GROUP>
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>

$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

To run the MPI example:

```console
$ mpirun -np 2 -bind-to none -map-by slot -- \
    python tsubame-optuna-example/tensorflow_mnist_eager_optuna.py $STUDY_NAME $STORAGE_URL
```

You can list the history of optimization as follows.
```console
$ python tsubame-optuna-example/print_study_history.py $STUDY_NAME $STORAGE_URL
```

## Distributed Optimization for ChainerMN with RDB

Let's parallelize a script written in ChainerMN.

At first, build the ChainerMN image and run a container:

```console
$ module load singularity/2.6.1
$ singularity pull docker://toshihikoyanase/chainer-ompi:latest
$ singularity shell --nv chainer-ompi-latest.simg
```

With the container, install Python dependencies under the user directory:

```console
$ pip install --user psycopg2-binary
# If you have already install pandas>=0.25.0, please downgrade it to 0.24.2 for Python 3.4.2.
$ pip uninstall pandas && pip install --user pandas==0.24.2
```

Similarly to the Horovod example, run the example with interactive node.

```console
$ GROUP=<YOUR_GROUP>
$ qrsh -g $GROUP -l h_node=1 -l h_rt=01:00:00
```

Create a study identifier in the container:

```console
$ singularity shell --nv chainer-ompi-latest.simg

$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/
$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

Run the example:

```console
$ mpirun -np 2 -bind-to none -map-by slot -- \
    python tsubame-optuna-example/chainermn_gpu.py $STUDY_NAME $STORAGE_URL
```

## Distributed Optimization for Tensorflow without RDB

Let's parallelize a script written in Horovod and TensorFlow.

Download MNIST data:

```console
$ wget -O ~/mnist.npz https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Here, we'll run the example with interactive node. (You also can consolidate the following commands as a batch job.)

```console
$ GROUP=<YOUR_GROUP>
$ qrsh -g $GROUP -l h_node=1 -l h_rt=01:00:00
```

Run a container:

```console
$ module load singularity/2.6.1
$ singularity shell --nv horovod-0.15.2-tf1.12.0-torch1.0.0-py3.5.simg
```

To run the MPI example:

```console
$ mpirun -np 2 -bind-to none -map-by slot -- \
    python tsubame-optuna-example/tensorflow_mnist_eager_optuna_inmemory.py
```


## Distributed Optimization for ChainerMN without RDB

Let's parallelize a script written in ChainerMN.

At first, build the ChainerMN image and run a container:

```console
$ module load singularity/2.6.1
$ singularity pull docker://toshihikoyanase/chainer-ompi:latest
$ singularity shell --nv chainer-ompi-latest.simg
```

With the container, install Python dependencies under the user directory:

```console
# If you have already install pandas>=0.25.0, please downgrade it to 0.24.2 for Python 3.4.2.
$ pip uninstall pandas && pip install --user pandas==0.24.2
```

Similarly to the Horovod example, run the example with interactive node.

```console
$ GROUP=<YOUR_GROUP>
$ qrsh -g $GROUP -l h_node=1 -l h_rt=01:00:00
```

Create a container:

```console
$ module load singularity/2.6.1
$ singularity shell --nv chainer-ompi-latest.simg
```

Run the example:

```console
$ mpirun -np 2 -bind-to none -map-by slot -- \
    python tsubame-optuna-example/chainermn_gpu_inmemory.py
```

## Distributed Optimization for Tensorflow + Horovod without singularity

Create `bin/setup_horovod.sh` to load modules necessary for TensorFlow + Horovod:

```console
#!/bin/bash
module load python/3.6.5
module load cuda/9.2.148
module load cudnn/7.4
module load nccl/2.4.2
module load openmpi/2.1.2-opa10.9-t3-thread-multiple
```

Setup Python modules using `venv`.
- A computing node with a GPU is required to setup modules correctly.
- `horovod==0.13.11` works with `openmpi/2.1.2`, but `horovod>=0.14.0` doesn't.

```console
$ qrsh -l s_gpu=1 -l h_rt=0:10:00
$ source bin/setup_horovod.sh
$ python3.6 -m venv venv-horovod
$ source venv-horovod/bin/activate
$ python3.6 -m pip install -U pip
$ pip install tensorflow-gpu
$ pip install horovod==0.13.11
$ pip install mpi4py
$ pip install git+https://github.com/pfnet/optuna.git@titech-horovod-examples
```

Create `scripts/horovod-example.sh` as a job file:

```console
#!/bin/bash
#$ -cwd
#$ -N horovod-example

. /etc/profile.d/modules.sh
. bin/setup_horovod.sh
source venv-horovod/bin/activate

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/horovod_gpu_inmemory.py
```

Submit the job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/horovod-example.sh
```


## Distributed Optimization for ChainerMN without singularity

Create `bin/setup_chainermn.sh` to load modules necessary for ChainerMN:

```console
#!/bin/bash
module load python/3.6.5
module load cuda/9.2.148
module load cudnn/7.4
module load nccl/2.4.2
module load openmpi/2.1.2-opa10.9-t3
```

Setup Python modules using `venv`.
- A computing node with a GPU is required to setup modules correctly.
- Please make sure that you select `cupy-cuda92` instead of `cupy` to reduce installation time.

```console
$ qrsh -l s_gpu=1 -l h_rt=0:10:00
$ source bin/setup_chainermn.sh
$ python3.6 -m venv venv-chainer
$ source venv-chainer/bin/activate
$ python3.6 -m pip install -U pip
$ pip install cupy-cuda92
$ pip install chainer
$ pip install mpi4py
$ pip install git+https://github.com/pfnet/optuna.git@titech-horovod-examples
```

Create `scripts/chainermn-example.sh` as a job file:

```console
#!/bin/bash
#$ -cwd
#$ -N flatmpi

. /etc/profile.d/modules.sh
. bin/setup_chainermn.sh
source venv-chainer/bin/activate

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/chainermn_gpu_inmemory.py
```

Submit the job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/chainermn-example.sh
```

## See Also

- [Optuna Tutorial](https://optuna.readthedocs.io/en/latest/tutorial/)
- [Optuna Examples](https://github.com/pfnet/optuna/tree/master/examples)
