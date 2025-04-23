#!/usr/bin/env bash
python receiver_decode.py \
  --model        conv_lstm_PNC32_ae \
  --model_path   conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth \
  --ip      127.0.0.1 \
  --port    9000 \
  --lstm_kwargs  '{"total_channels":32,"hidden_channels":32,"ae_model_name":"PNC32","bidirectional":false}' \
  --deadline_ms 100 \
  # --quant False


##### copy from below for ease of access:
# pnc32
# PNC32_final_w_taildrops.pth
# conv_lstm_PNC32_ae
# conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth
# conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth