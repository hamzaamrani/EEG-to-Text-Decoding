python3 eval_decoding_raw.py \
    --checkpoint_path ./checkpoints/decoding_raw_104_h/last/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b4_25_25_5e-05_5e-05_unique_sent_best.pt \
    --config_path ./config/decoding_raw/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b4_25_25_5e-05_5e-05_unique_sent.json \
    -cuda cuda:0

