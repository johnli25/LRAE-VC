# LRAE-VC

## Commands, Instructions, Setup
- Setting up server, Conda, and GPUs (based on UIUC NCSA's Delta server): https://docs.google.com/document/d/1U5KpvcJr5ousA-zq9EcdzArJlSgpgM4wdYXXYV6tCLg/edit?tab=t.0
#### Training
- To run PNC autoencoder (no classification) `python autoencoder_train.py --model=PNC`
- To run PNC autoencoder (with classification integrated) `python autoencoder_train.py --model=PNC_with_classification`
- To run LRAE_VC autoencoder (no classification) `python autoencoder_train.py --model=LRAE_VC`
#### Testing/Inference
- To run the simple + assumed to be "pretrained" classifier: ``
#### Sender + Receiver Test & Simulation
- **Run receiver first!** via (below is example, change args as necessary)
```
python receiver_decode.py --model_path="PNC_final_w_random_drops.pth" --host=127.0.0.1 --port=8080
```
and wait till it says "Listening on ..."
- Then run sender (below is example, change args as necessary):
```
python sender_encode.py --input_dir="UCF_224x224x3_PNC_FrameCorr_input_imgs/" --model_path="PNC_final_w_random_drops.pth" --host=127.0.0.1 --port=8080
```

#### Team: John Li, AJ Grama, Wenjie Guo

### General TODOs

- [x] ~~Implement the second and dual neural network for PREDICTING missing features in latent encodings~~
- [x] ~~Implement and incorporate object classification into the autoencoder NN. How to do this? --> (https://docs.google.com/document/d/1svHaRZ1yiAsARJDC_MInBo5Ln_tRjIg14dhBlCV_UsI/edit?usp=sharing)~~

### Specific TODOs + Issues Log

### Notes + Other References:
- Results sheet: https://docs.google.com/spreadsheets/d/1NVdFgHwTFBAl3Qp2PYW8EZE4UFLQ0xObCSjxnE2KDeo/edit?usp=sharing

### Dev Log
- NOTE: If you run `python autoencoder_train.py --model="PNC_NoTail"` any `max_tail_len`, `tail_length`, and related vars will be technically misleading. I didn't feel like updating the variable names in the code b/c lazy 😆, but functionally, PNC_NoTail should be fine.
