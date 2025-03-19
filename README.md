# LRAE-VC

## Commands, Instructions, Setup
- Setting up server, Conda, and GPUs (based on UIUC NCSA's Delta server): https://docs.google.com/document/d/1U5KpvcJr5ousA-zq9EcdzArJlSgpgM4wdYXXYV6tCLg/edit?tab=t.0
#### Training
- To run PNC autoencoder (no classification) `python autoencoder_train.py --model=PNC`
- To run PNC autoencoder (with classification integrated) `python autoencoder_train.py --model=PNC_with_classification`
- To run LRAE_VC autoencoder (no classification) `python autoencoder_train.py --model=LRAE_VC`
- To run conv_lstm_ae (whose current AE baseline is PNC16) with NO dropped out features: `python autoencoder_train_vid_sequence.py --model conv_lstm_ae`
- To run conv_lstm_ae (whose current AE baseline is PNC16) with up to x/16 zeroed out features dropped out: `python autoencoder_train_vid_sequence.py --model conv_lstm_ae --model_path conv_lstm_ae_final_weights.pth --epochs 25 --drops 12` (e.g. x = 12 here)
- 
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

### TODOs

- [x] ~~Implement the second and dual neural network for PREDICTING missing features in latent encodings~~
- [x] ~~Implement and incorporate object classification into the autoencoder NN. How to do this? --> (https://docs.google.com/document/d/1svHaRZ1yiAsARJDC_MInBo5Ln_tRjIg14dhBlCV_UsI/edit?usp=sharing)~~

### Notes + Other References:
- Results sheet: https://docs.google.com/spreadsheets/d/1NVdFgHwTFBAl3Qp2PYW8EZE4UFLQ0xObCSjxnE2KDeo/edit?usp=sharing

### Dev Log + Miscellaneous System CMDs:
- just do `ssh gpua058` in another terminal to access from another termina
- `nvidia-smi` for GPU stats
- `du -sh path_to_file_or_folder` to show disk space of folder or file
- `ls -1 /path/to/directory | wc -l` to count the number of files in a directory
- `ls -1 /path/to/directory | wc -l` to count the number of files in a directory
- ** YOU MUST do `srun --pty bash`, `ssh gpuaXXX`, or in general be inside the GPU VM to export env vars MASTER_ADDR and MASTER_PORT for PyTorch's DDP to work! `export MASTER_ADDR=$(hostname)` and `export MASTER_PORT=12355` and then run w/ `python your_script.py` !!
- Make sure to ONLY do `python your_script.py` when you're in the GPU VM via `srun --pty bash`
- OTHERWISE, do `srun python your_script.py` if you're OUTSIDE of the GPU VM.
- If you're using PyTorch's DDP: `srun python -m torch.distributed.launch your_script.py`

### What I learned + Conceptual stuff
##### What Is Redundancy?

**Redundancy** means that the same critical information is stored in more than one place. In the context of an autoencoder’s latent space, redundancy means that even if some of the features (or channels) are lost or dropped, the remaining features still contain enough information to allow the decoder to reconstruct the original input accurately.

---

##### Why Might a Model Learn Redundancy with Tail Dropout but Not as Effectively with Random Interspersed Dropouts?

**Tail Dropout Strategy:**
- **Ordered Information:**  
  When you always drop features at the tail (the last few channels), the network quickly learns that the first few channels are almost always available. It then places the most critical, base-level information into these early channels.  
- **Incentive to Be Redundant:**  
  Since the tail is frequently dropped, the model is forced to duplicate or shift important information into the base channels. This **built-in redundancy** ensures that even if the enhancement layers (the tail) are missing, the core information needed for reconstruction is still preserved.

**Random Interspersed Dropout Strategy:**
- **Uniform Randomness:**  
  With random interspersed dropout, any channel—regardless of its position—can be dropped during training. There is no consistent pattern.  
- **No Clear “Safe Zone”:**  
  Because the dropout is unpredictable across all channels, the network cannot rely on a subset of channels (like the early ones) to always be present. This makes it harder for the network to learn an ordered, redundant structure where some channels are reserved as a robust base.  
- **Distributed Representation:**  
  The network ends up learning to spread information more evenly across all channels rather than concentrating critical details into a protected subset. While this does enforce some redundancy (since every channel must potentially compensate for a missing one), it doesn’t create a clear hierarchy of “base” versus “enhancement” features. This can make the network more vulnerable when a crucial channel is dropped, as there's no predictable backup for the information.

---
In Summary

- **Redundancy** is about duplicating key information so that loss of some parts doesn’t cripple performance.
- **Tail dropout** encourages the network to concentrate essential information into the early channels because those channels are reliably available. This promotes redundancy in a progressive, ordered manner.
- **Random interspersed dropout** applies uniformly across all channels, forcing the network to spread information evenly rather than creating a reliable “base layer.” As a result, it may not foster the same type of redundancy where some channels are reliably preserved. 

The choice between the two strategies depends on your design goals. If you want a progressive representation where some features are consistently available (mimicking a base layer in progressive transmission), tail dropout is more effective. If you aim to simulate completely random loss, interspersed dropout is closer to reality—but it might not lead to as robust an ordering of information unless you design additional mechanisms to encourage redundancy.
