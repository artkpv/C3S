set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

function vast() {
    if [[ "${VAST_CONTAINERLABEL:-}" != "" ]] ; then
        # At Vast.
        conda update conda
        conda env update -n base -f environment.yml
    fi
}

if [[ $# != 0 ]] ; then 
    cmd=$1
    shift 1
    $cmd "$@"  
fi
