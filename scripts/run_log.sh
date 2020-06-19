# Peek into argument list and COPY OUT experiment and video names
ARGS=""
while [[ $# -gt 0 ]]
do
    case $1 in
        --exp-name)
            EXP_NAME="$2"
            ARGS="$ARGS $1"
            shift
            ;;
        --video-path)
            VIDEO_NAME=$(basename $2 .mp4)
            ARGS="$ARGS $1"
            shift
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# Output directory to save log.txt
OUTPUT_DIR="dod/output/DOD/$VIDEO_NAME/$EXP_NAME"

# Determine log name.
# We don't want existing logs to be overwritten.
i=1
while [[ -f "$OUTPUT_DIR/log$i.txt" ]]
do
    i=$(( $i + 1 ))
done
echo "Log will be saved to $OUTPUT_DIR/log$i.txt"
    
# Create output directory, if doesn't already exist
mkdir -p "$OUTPUT_DIR"

# Run system
mpirun -H jetson,localhost bash mpi.sh $ARGS 2>&1 | tee -a "$OUTPUT_DIR/log$i.txt"
