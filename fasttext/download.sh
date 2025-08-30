group_id=${1:-0}
source /data/anaconda3/bin/activate base
python /data/code/Oprover/fasttext/download.py --threads 256 --processes 64 --group_id $group_id