set -e
# frontal only no clahe no oversample only decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_no_clahe_no_oversample_decoder"\
    --model_mode "encoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal"

# frontal-lateral only no clahe no oversample only decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_lateral_no_clahe_no_oversample_decoder"\
    --model_mode "encoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal_and_lateral"


# frontal only no clahe no oversample encoder-decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_no_clahe_no_oversample_encoder_decoder"\
    --model_mode "encoder-decoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal"

# frontal-lateral only no clahe no oversample encoder-decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_lateral_no_clahe_no_oversample_encoder_decoder"\
    --model_mode "encoder-decoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal_and_lateral"

# # frontal-lateral no clahe oversample encoder-decoder
accelerate launch -m train.train\
    --train_csv "train_resized.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_lateral_no_clahe_oversample_encoder_decoder"\
    --model_mode "encoder-decoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal_and_lateral"

# # frontal-lateral clahe oversample encoder-decoder
accelerate launch -m train.train\
    --train_csv "train_resized.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_lateral_clahe_oversample_encoder_decoder"\
    --model_mode "encoder-decoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal_and_lateral"
    --clahe

# # frontal only clahe no oversample only decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_clahe_no_oversample_decoder"\
    --model_mode "encoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal" --clahe

# frontal-lateral clahe no oversample only decoder
accelerate launch -m train.train\
    --train_csv "train_resized_v0.csv"\
    --test_csv "test_resized.csv"\
    --train_name "frontal_lateral_clahe_no_oversample_decoder"\
    --model_mode "encoder"\
    --output_dir "checkpoints"\
    --image_mode "frontal_and_lateral" --clahe