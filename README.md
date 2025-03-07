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

### Dev Log + Misc System CMDs:
- just do `ssh gpua058` in another terminal to access from another termina
- `nvidia-smi` for GPU stats
- `du -sh path_to_file_or_folder` to show disk space of folder or file
- ** YOU MUST do `srun --pty bash`, `ssh gpuaXXX`, or in general be inside the GPU VM to export env vars MASTER_ADDR and MASTER_PORT for PyTorch's DDP to work! `export MASTER_ADDR=$(hostname)` and `export MASTER_PORT=12355` and then run w/ `python your_script.py` !!
- Make sure to ONLY do `python your_script.py` when you're in the GPU VM via `srun --pty bash`
- OTHERWISE, do `srun python your_script.py` if you're OUTSIDE of the GPU VM.
- If you're using PyTorch's DDP: `srun python -m torch.distributed.launch your_script.py`
