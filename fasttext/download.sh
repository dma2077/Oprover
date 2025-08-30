group_id=${1:-0}
source /data/anaconda3/bin/activate base
eval $(curl -s http://deploy.i.basemind.com/httpproxy)
python /data/code/Oprover/fasttext/download.py --threads 128 --processes 32 --group_id $group_id