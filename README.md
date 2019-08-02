# Optuna Examples for TSUBAME3.0

This is a tutorial material to use Optuna in the [TSUBAME3.0](https://www.t3.gsic.titech.ac.jp/) infrastructure (unofficial).

This tutorial describes:

- Minimum setup to run Optuna.
- How to launch Optuna storage on an interactive node.
- How to parallelize single node ML training.
- How to parallelize multi-node, MPI-based ML training.

## Minimum Setup of Optuna in TSUBAME

The following example provides quickstart of Optuna.

Points
- Optuna can easily installed by `pip`.
- `sqlite:///example.db` is an RDB URL to specify the storage of optimization results. In this case, `SQLite` is specified.
- You can use `PostgreSQL` or in-memory storage instead of `SQLite`.


```console
$ qrsh -l s_core=1 -l h_rt=00:10:00
$ module load python/3.6.5
$ pip install --user optuna
$ python tsubame-optuna-example/quadratic.py quickstart sqlite:///example.db
$ python tsubame-optuna-example/print_study_history.py quickstart sqlite:///example.db
```

## Launch PostgreSQL in TSUBAME

RDB servers can be used for parallel optimization.
In this tutorial, we use PostgreSQL.

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

## Distributed Optimization for Single Node Learning

Let's parallelize a simple Optuna script that optimizes a quadratic function.

Set up the RDB URL and create a study identifier:

```console
$ STORAGE_HOST=<HOST_WHERE_POSTGRES_IS_RUNNING>
$ STORAGE_URL=postgres://postgres@$STORAGE_HOST:5432/

$ module load python/3.6.5
$ pip install --user psycopg2-binary
$ STUDY_NAME=`~/.local/bin/optuna create-study --storage $STORAGE_URL`
```

Set up a shell script for qsub command, e.g.:

```console
$ echo "module load python/3.6.5" >> run_quadratic.sh
$ echo "python tsubame-optuna-example/quadratic.py $STUDY_NAME $STORAGE_URL" >> run_quadratic.sh
```

You can parallelize the optimization just by submitting multiple jobs.
For example, the following commands simultaneously run three workers in a study.

```console
$ GROUP=<YOUR_GROUP>

$ qsub -g $GROUP -l s_core=1 run_quadratic.sh
$ qsub -g $GROUP -l s_core=1 run_quadratic.sh
$ qsub -g $GROUP -l s_core=1 run_quadratic.sh
```

You can list the history of optimization as follows.
```console
$ python tsubame-optuna-example/print_study_history.py $STUDY_NAME $STORAGE_URL
```

## Optimize ChainerMN

See [this document](./README.chainermn.md).

## Optimize TensorFlow + Horovod

See [this document](./README.tensorflow.md).

## See Also

- [Optuna Tutorial](https://optuna.readthedocs.io/en/latest/tutorial/)
- [Optuna Examples](https://github.com/pfnet/optuna/tree/master/examples)
