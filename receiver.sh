#!/usr/bin/env bash
python receiver_decode.py \
  --model        pnc32 \
  --model_path   PNC32_final_w_taildrops_quantize256_8bits.pth \
  --ip      0.0.0.0 \
  --port    9000 \
  --deadline_ms 1300 \
  --quant


##### copy from below for ease of access:
# pnc32
# PNC32_final_w_taildrops.pth
# PNC32_final_no_dropouts_quantize256_8bits.pth
# PNC32_final_w_taildrops_quantize256_8bits.pth