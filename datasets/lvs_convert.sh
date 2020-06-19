# Convert videos to 720p
folder="$(dirname "$1")"
file="$(basename -- "$1")"
name="${file%.*}"
ext="${file##*.}"
conv="$name-720p.$ext"

cd $folder
ffmpeg -i "$file" -s 1280x720 "$conv"
