#!/usr/bin/env bash
# stdbuf -oL -eL 
python3 mininet_receiver.py \
  --model        pnc32 \
  --model_path   ../PNC32_final_w_taildrops.pth \
  --ip           0.0.0.0 \
  --port         9000 \
  --deadline_ms  22.7 \
  # > receiver_log.txt 2>&1

  #--lstm_kwargs  '{"total_channels":32,"hidden_channels":32,"ae_model_name":"PNC32","bidirectional":false}' \
  # --quant False

##### copy from below for ease of access:
# pnc32
# PNC32_final_w_taildrops.pth
# conv_lstm_PNC32_ae
# conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth
# conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth