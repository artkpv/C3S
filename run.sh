set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

function vast() {
    if [[ "$VAST_CONTAINERLABEL" != "" ]] ; then
        # At Vast.
        conda env create -f environment.yml
    else
        VAST_ID=`vast.py show instances | head -2 | tail -1 | cut -d ' ' -f`
        if [[ "$VAST_ID" != "" ]] ; then 
            vast.py execute $VAST_ID \
                git clone https://github.com/artkpv/DLK-works.git  --branch dev --recurse-submodules
            vast.py execute $VAST_ID 'cd DLK-works ; bash run.sh vast'
        fi
    fi
}

if [[ $# != 0 ]] ; then 
    cmd=$1
    shift 1
    $cmd "$@"  
fi
