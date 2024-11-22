# LRAE-VC

## Commands, Instructions, Setup
- Setting up server, Conda, and GPUs (based on UIUC NCSA's Delta server): https://docs.google.com/document/d/1U5KpvcJr5ousA-zq9EcdzArJlSgpgM4wdYXXYV6tCLg/edit?tab=t.0 
- To run PNC autoencoder (no classification) `python autoencoder_train.py --model=PNC`
- To run PNC autoencoder (with classification integrated) `python autoencoder_train.py --model=PNC_with_classification`
- To run LRAE_VC autoencoder (no classification) `python autoencoder_train.py --model=LRAE_VC`

#### Team: John Li, AJ Grama, Wenjie Guo

### General TODOs

- [ ] Implement the second and dual neural network for PREDICTING missing features in latent encodings
- [ ] Implement and incorporate object classification into the autoencoder NN. How to do this? --> (https://docs.google.com/document/d/1svHaRZ1yiAsARJDC_MInBo5Ln_tRjIg14dhBlCV_UsI/edit?usp=sharing)
- [ ] Experimental setup with random packet drops and deadline misses (much easier said than done)
- [ ] Lots of overall architecture tuning

### Specific TODOs + Issues Log
- [x] ~~Fix PNC dimension error~~
- [x] ~~Add percentage of labels classified correctly metric at the end~~
- [ ] Run/Train train_combined_model and fix any issues
- [ ] Train a separate classification model on the DECODED images (which are assumed to be on server side btw. No need to run encoder/autoencoder for this!)
- [ ] Run/Train AJ's LSTM-feature filling code (get_features.py and feature_filling.py).
- [ ] EXPERIMENT TASK: Create and implement sender_encode.py to send data back and forth btwn sender and receiver for LRAE-VC (no drops/misses)
- [ ] EXPERIMENT TASK: Send data back and forth btwn sender and receiver for PNC (**WITH** random packet drops + deadline messes. Can just manually drop them from the sender if you wanna "cheat" instead of setting up any actual networking LOL. 
- [ ] EXPERIMENT TASK: Same thing as above but for LRAE-VC (MAKE SURE TO USE LSTM-BASED FEATURE FILLING MODEL!) 

### Notes + Other References:
- Results sheet: https://docs.google.com/spreadsheets/d/1NVdFgHwTFBAl3Qp2PYW8EZE4UFLQ0xObCSjxnE2KDeo/edit?usp=sharing

### Dev Log
