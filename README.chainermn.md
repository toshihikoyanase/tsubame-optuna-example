# Optuna Examples for ChainerMN

This tutorial describes:

- How to tune ChainerMN training.
- How to use RDB to persist tuning results.


## Tuning ChainerMN Training

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
- Please refer to [this document](https://gist.github.com/keisukefukuda/a260d00c62c53811272ff83daf400dee#file-0-readme-md) for further details of ChainerMN setup (in Japanese).

```console
$ qrsh -l s_gpu=1 -l h_rt=0:10:00
$ source bin/setup_chainermn.sh
$ python3.6 -m venv venv-chainer
$ source venv-chainer/bin/activate
$ python3.6 -m pip install -U pip
$ pip install mpi4py cupy-cuda92==6.2.0 chainer==6.2.0
$ pip install git+https://github.com/pfnet/optuna.git@titech-horovod-examples
```

Create `scripts/chainermn-example.sh` as a job file:

```console
#!/bin/bash
#$ -cwd
#$ -N chainermn

. /etc/profile.d/modules.sh
. bin/setup_chainermn.sh
source venv-chainer/bin/activate

# OMP_NUM_THREADS affects the performance if you use Chainer with OpenCV.
export OMP_NUM_THREADS=1

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/chainermn_mnist_inmemory.py
```

Submit the job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/chainermn-example.sh
```


## Use RDB to Persist Tuning Results

In this example, we use PostgreSQL to save/load studies.
Please note that `sqlite` may not work correctly if you use it on NFS.

At first, please launch PostgreSQL using singularity. See [this document](./README.md).

Install RDB driver.

```console
$ qrsh -l s_gpu=1 -l h_rt=0:10:00
$ source bin/setup_chainermn.sh
$ source venv-chainer/bin/activate
$ pip install psycopg2-binary
```

Create a study.

```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/

$ STUDY_NAME=`optuna create-study --storage $STORAGE_URL`
```

Create `scripts/chainermn-db-example.sh` as a job script:

```console
#!/bin/bash
#$ -cwd
#$ -N chainermn-db

. /etc/profile.d/modules.sh
. bin/setup_chainermn.sh
source venv-chainer/bin/activate

# OMP_NUM_THREADS affects the performance if you use Chainer with OpenCV.
export OMP_NUM_THREADS=1

mpirun -npernode 1 -n 2 -x PATH  -x LD_LIBRARY_PATH \
    -- python tsubame-optuna-example/chainermn_mnist.py \
    (put the study name as a string here) \
    (put the storage url as a string here)
```

Submit a job:

```console
$ qsub -l s_gpu=2 -l h_rt=00:10:00 ./scripts/chainermn-db-example.sh
```
