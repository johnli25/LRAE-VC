#!/usr/bin/env bash
python sender_encode.py \
  --model        conv_lstm_PNC32_ae \
  --model_path   conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth \
  --input        ./TUCF_sports_action_224x224_mp4_vids \
  --ip      127.0.0.1 \
  --port    9000 \
  --lstm_kwargs  '{"total_channels":32,"hidden_channels":32,"ae_model_name":"PNC32","bidirectional":false}' \
  # --quant False


##### copy from below for ease of access:
# pnc32
# PNC32_final_w_taildrops.pth
# conv_lstm_PNC32_ae
# conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth
# conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth