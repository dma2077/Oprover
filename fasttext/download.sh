group_id=$1
source /data/anaconda3/bin/activate
python /data/code/Oprover/fasttext/download.py --threads 16 --processes 64 --group_id $group_id