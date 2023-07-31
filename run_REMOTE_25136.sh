set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

function vast() {
    if [[ "${VAST_CONTAINERLABEL:-}" != "" ]] ; then
        # At Vast.
        conda update conda
        conda install --freeze-installed \
            $( cat requirements.txt | grep -v ' # pip' ) \
            -c conda-forge -c pytorch -c r -c defaults
        pip install $( cat requirements.txt | sed -En '/# pip/s_^(.*) # pip_\1_p'  ) 
    else
        VAST_ID=`vast.py show instances | head -2 | tail -1 | cut -d ' ' -f1 `
        if [[ "$VAST_ID" == "" ]] ; then 
            VAST_ID=` vast.py search offers 'reliability > 0.9 disk_space > 80 gpu_ram > 16 cpu_cores >= 6 rentable=true verified=true' -o 'dph' | head -2 | tail -1 | cut -d ' ' -f1 `
            vast.py create instance $VAST_ID --image pytorch/pytorch --disk 50
        fi
        SSHURL=` vast.py ssh-url $VAST_ID `
        echo running at $SSHURL
        ssh $SSHURL git clone https://github.com/artkpv/DLK-works.git  --branch dev --recurse-submodules
        ssh $SSHURL 'cd DLK-works ; bash run.sh vast'
    fi
}

if [[ $# != 0 ]] ; then 
    cmd=$1
    shift 1
    $cmd "$@"  
fi
