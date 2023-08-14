echo "This scirpt construct .pickle files from .mat files from ZuCo dataset with raw EEG signals."
echo "This script also generates tenary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

python3 ./util/construct_dataset_mat_to_pickle_v1_withRaw.py -t task1-SR
python3 ./util/construct_dataset_mat_to_pickle_v1_withRaw.py -t task2-NR
python3 ./util/construct_dataset_mat_to_pickle_v1_withRaw.py -t task3-TSR
python3 ./util/construct_dataset_mat_to_pickle_v2_withRaw.py

python3 ./util/get_sentiment_labels.py
python3 ./util/get_SST_ternary_dataset.py
