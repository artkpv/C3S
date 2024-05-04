set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

function train() {
    if ! command -v sbatch &> /dev/null; then
        python train.py "$@"
    else
        _slurm train.py "$@"
    fi
}

function eval() {
    if ! command -v sbatch &> /dev/null; then
        python eval.py "$@"
    else
        _slurm eval.py "$@"
    fi
}

function train_eval() {
    if ! command -v sbatch &> /dev/null; then
        python train_eval.py "$@"
    else
        _slurm train_eval.py "$@"
    fi
}

function experiment() {
    for i in {1..5}; do
        if ! command -v sbatch &> /dev/null; then
            python train_eval.py "$@"
        else
            timestamp=$( date +%Y%m%d_%H-%M-%S )
            sbatch --export=ALL --output=artifacts/slurm-%j-${timestamp}.out launch.batch train_eval.py "$@"
            sleep 2
        fi
    done
}

function report() {
    timestamp=$( date +%Y%m%d_%H-%M-%S )
    sbatch --export=ALL --gpus-per-node=0 --output=artifacts/slurm-%j-${timestamp}.out launch.batch report.py "$@"
    echo 'Waiting job...'
    # Wait till output file appears and then print it
    while [ ! -f artifacts/slurm-*-${timestamp}.out ]; do sleep 1; done
    less artifacts/slurm-*-${timestamp}.out
}

function _slurm() {
    timestamp=$( date +%Y%m%d_%H-%M-%S )
    sbatch --export=ALL --output=artifacts/slurm-%j-${timestamp}.out launch.batch "$@"
    echo 'Waiting job...'
    # Wait till output file appears and then print it
    while [ ! -f artifacts/slurm-*-${timestamp}.out ]; do sleep 1; done
    less artifacts/slurm-*-${timestamp}.out
}

if [[ $# != 0 ]] ; then 
    cmd=$1
    shift 1
    $cmd "$@"  
fi

if [[ $# == 0 ]] ; then 
    echo "No command provided. Exiting."
else 
    cmd=$1
    shift 1
    $cmd "$@"
fi
