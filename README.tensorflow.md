# Optuna Examples for TensorFlow + Horovod

This tutorial describes:

- How to tune TensorFlow + Horovod training.
- How to use RDB to persist tuning results.


## Tuning TensorFlow + Horovod Training

Create `bin/setup_tensorflow.sh` to load modules necessary for TensorFlow + Horovod:

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
$ source bin/setup_tensorflow.sh
$ python3.6 -m venv venv-tensorflow
$ source venv-tensorflow/bin/activate
$ python3.6 -m pip install -U pip
$ pip install tensorflow-gpu
$ pip install horovod==0.13.11
$ pip install mpi4py
$ pip install git+https://github.com/pfnet/optuna.git@titech-horovod-examples
```

Create `scripts/tensorflow-example.sh` as a job script:

```console
#!/bin/bash
#$ -cwd
#$ -N tensorflow-example

. /etc/profile.d/modules.sh
. bin/setup_tensorflow.sh
source venv-tensorflow/bin/activate

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/tensorflow_mnist_inmemory.py
```

Submit a job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/tensorflow-example.sh
```


## Use RDB to Persist Tuning Results

In this example, we use PostgreSQL to save/load studies.
Please note that `sqlite` may not work correctly if you use it on NFS.

At first, please launch PostgreSQL using singularity. See [this document](./README.md).

Install RDB driver.

```console
$ qrsh -l s_gpu=1 -l h_rt=0:10:00
$ source bin/setup_tensorflow.sh
$ source venv-tensorflow/bin/activate
$ pip install psycopg2-binary
```

Create a study.

```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/

$ STUDY_NAME=`optuna create-study --storage $STORAGE_URL`
```

Create `scripts/tensorflow-db-example.sh` as a job script:

```console
#!/bin/bash
#$ -cwd
#$ -N tensorflow-db

. /etc/profile.d/modules.sh
. bin/setup_tensorflow.sh
source venv-tensorflow/bin/activate

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/tensorflow_mnist.py \
    (put the study name as a string here) \
    (put the storage url as a string here)
```

Submit a job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/tensorflow-db-example.sh
```
