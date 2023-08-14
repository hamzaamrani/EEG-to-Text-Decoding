# Deep Representation Learning for Open Vocabulary Electroencephalography-to-Text Decoding

## Create Environment
run `conda env create -f environment.yml` to create the conda environment (named "EEGToTextOpenVoc") used in our experiments.

## Download ZuCo datasets
- Download ZuCo v1.0 'Matlab files' for 'task1-SR','task2-NR','task3-TSR' from https://osf.io/q3zws/files/ under 'OSF Storage' root,  
unzip and move all `.mat` files to `/dataset/ZuCo/task1-SR/Matlab_files`,`/dataset/ZuCo/task2-NR/Matlab_files`,`/dataset/ZuCo/task3-TSR/Matlab_files` respectively.
- Download ZuCo v2.0 'Matlab files' for 'task1-NR' from https://osf.io/2urht/files/ under 'OSF Storage' root, unzip and move all `.mat` files to `/dataset/ZuCo/task2-NR-2.0/Matlab_files`.

## Preprocess datasets
run `bash ./scripts/prepare_dataset_raw.sh` to preprocess `.mat` files and prepare sentiment labels. 

For each task, all `.mat` files will be converted into one `.pickle` file stored in `/dataset/ZuCo/<task_name>/<task_name>-dataset.pickle`. 

## Usage Example
### Open vocabulary EEG-To-Text Decoding
To train an EEG-To-Text decoding model, run `bash ./scripts/train_decoding_raw.sh`.

To evaluate the trained EEG-To-Text decoding model from above, run `bash ./scripts/eval_decoding_raw.sh`.
