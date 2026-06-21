ROOT="checkpoints"
TEST_CSV="test_resized.csv"
  
  # frontal, no clahe, encoder-decoder
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-decoder-frontal-frontal_no_clahe_no_oversample_encoder_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_no_clahe_no_oversample_encoder_decoder.csv" \
#   --image_mode "frontal"

# # frontal+lateral, clahe, oversample, encoder-decoder
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-decoder-frontal_and_lateral-frontal_lateral_clahe_oversample_encoder_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_lateral_clahe_oversample_encoder_decoder.csv" \
#   --image_mode "frontal_and_lateral" \
#   --apply_clahe_fn

# # frontal+lateral, no clahe, no oversample, encoder-decoder
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-decoder-frontal_and_lateral-frontal_lateral_no_clahe_no_oversample_encoder_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_lateral_no_clahe_no_oversample_encoder_decoder.csv" \
#   --image_mode "frontal_and_lateral"

# # frontal+lateral, no clahe, oversample, encoder-decoder
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-decoder-frontal_and_lateral-frontal_lateral_no_clahe_oversample_encoder_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_lateral_no_clahe_oversample_encoder_decoder.csv" \
#   --image_mode "frontal_and_lateral"

# # frontal, clahe, no oversample, encoder (decoder)
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-frontal-frontal_clahe_no_oversample_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_clahe_no_oversample_encoder.csv" \
#   --image_mode "frontal" \
#   --apply_clahe_fn

# # frontal, no clahe, no oversample, encoder (decoder)
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-frontal-frontal_no_clahe_no_oversample_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_no_clahe_no_oversample_encoder.csv" \
#   --image_mode "frontal"

# # frontal+lateral, clahe, no oversample, encoder (decoder)
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-frontal_and_lateral-frontal_lateral_clahe_no_oversample_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_lateral_clahe_no_oversample_encoder.csv" \
#   --image_mode "frontal_and_lateral" \
#   --apply_clahe_fn

# # frontal+lateral, no clahe, no oversample, encoder (decoder)
# python3 -m eval.generate_response \
#   --lora_path "$ROOT/qwen-vl-report-lora-encoder-frontal_and_lateral-frontal_lateral_no_clahe_no_oversample_decoder/final_lora" \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_frontal_lateral_no_clahe_no_oversample_encoder.csv" \
#   --image_mode "frontal_and_lateral"


# python3 -m eval.generate_response \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_baseline_frontal_lateral.csv" \
#   --image_mode "frontal_and_lateral" --max_new_tokens 300

# python3 -m eval.generate_response \
#   --test_csv "$TEST_CSV" \
#   --output_csv "results_baseline_frontal.csv" \
#   --image_mode "frontal" --max_new_tokens 300

ROOT="Indiana-CXR-Eden"
  
python -m eval.eval --input_csv "$ROOT/results_frontal_no_clahe_no_oversample_encoder_decoder.csv" --output_csv "green_score/results_frontal_no_clahe_no_oversample_encoder_decoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_lateral_clahe_oversample_encoder_decoder.csv" --output_csv "green_score/results_frontal_lateral_clahe_oversample_encoder_decoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_lateral_no_clahe_no_oversample_encoder_decoder.csv" --output_csv "green_score/results_frontal_lateral_no_clahe_no_oversample_encoder_decoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_lateral_no_clahe_oversample_encoder_decoder.csv" --output_csv "green_score/results_frontal_lateral_no_clahe_oversample_encoder_decoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_clahe_no_oversample_encoder.csv" --output_csv "green_score/results_frontal_clahe_no_oversample_encoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_no_clahe_no_oversample_encoder.csv" --output_csv "green_score/results_frontal_no_clahe_no_oversample_encoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_lateral_clahe_no_oversample_encoder.csv" --output_csv "green_score/results_frontal_lateral_clahe_no_oversample_encoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_frontal_lateral_no_clahe_no_oversample_encoder.csv" --output_csv "green_score/results_frontal_lateral_no_clahe_no_oversample_encoder.csv" --green

python -m eval.eval --input_csv "$ROOT/results_baseline_frontal_lateral.csv" --output_csv "green_score/results_baseline_frontal_lateral.csv" --green

python -m eval.eval --input_csv "$ROOT/results_baseline_frontal.csv" --output_csv "green_score/results_baseline_frontal.csv" --green