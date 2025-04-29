#!/usr/bin/env bash
python receiver_decode.py \
  --model        pnc32 \
  --model_path   PNC32_final_no_dropouts_quantize256_8bits.pth \
  --ip      0.0.0.0 \
  --port    9000 \
  --deadline_ms 1300 \
  # --lstm_kwargs  '{"total_channels":32,"hidden_channels":32,"ae_model_name":"PNC32","bidirectional":false}' \
  --quant


##### copy from below for ease of access:
# pnc32
# PNC32_final_w_taildrops.pth
# conv_lstm_PNC32_ae
# conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth
# conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth
# conv_lstm_PNC32_ae_dropUpTo_32_quantize256_8_bits.pth
# PNC32_final_no_dropouts_quantize256_8bits.pth
# conv_lstm_PNC32_ae_dropUpTo_32_quantize256_8_bits.pth