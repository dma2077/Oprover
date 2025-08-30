group_id=${1:-0}
source /data/anaconda3/bin/activate base
python /data/code/Oprover/fasttext/download.py --threads 512 --processes 1 --group_id $group_id