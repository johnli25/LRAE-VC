import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
pnc_df = pd.read_csv("PNC_results.csv")
castr_df = pd.read_csv("CASTR_results.csv")
grace_df = pd.read_csv("../Grace/results/grace/all.csv")

# Manual mapping
tail_to_percent = {
    0: 0.0,
    3: 10.0,
    6: 20.0,
    10: 30.0,
    13: 40.0,
    16: 50.0,
    19: 60.0,
    22: 70.0,
    26: 80.0,
    28: 88.0
}

# Map the drop percent
pnc_df["loss"] = pnc_df["tail_len_drop"].map(tail_to_percent)
castr_df["loss"] = castr_df["tail_len_drop"].map(tail_to_percent)

# Separate CASTR into 1/3/5 consecutive drops
castr_1 = castr_df[castr_df["consecutive"] == 1]
castr_3 = castr_df[castr_df["consecutive"] == 3]
castr_5 = castr_df[castr_df["consecutive"] == 5]

# Filter GRACE
grace_df_base = grace_df[
    ((grace_df['model_id'] == 1024) &
     (grace_df['video'].isin([
         'diving7_224x224.mp4',
        #  'Golf-Swing-Front005_224x224.mp4',
        #  "Kicking-Front003_224x224.mp4",
        #  "Lifting002.mp4",
        #  "Riding-Horse006.mp4",
        #  "Run-Side001.mp4",
        #  "SkateBoarding-Front003.mp4",
        #  "Swing-Bench016.mp4",
        #  "Swing-SideAngle006.mp4",
        #  "Walk-Front021.mp4",
     ]))) &
    (grace_df['frame_id'] != 0)
]

grace_1 = grace_df_base[grace_df_base['nframes'] == 1]
grace_3 = grace_df_base[grace_df_base['nframes'] == 3]
grace_5 = grace_df_base[grace_df_base['nframes'] == 5]

grace_1_mse = grace_1.groupby('loss')['mse'].mean()
# grace_1_mse.index = (grace_1_mse.index * 100).round()
grace_1_psnr = 10 * np.log10(1.0 / grace_1_mse)
grace_1_psnr.index = (grace_1_psnr.index * 100).round()
grace_1_ssim = grace_1.groupby('loss')['ssim'].mean()
grace_1_ssim.index = (grace_1_ssim.index * 100).round()

grace_3_mse = grace_3.groupby('loss')['mse'].mean()
grace_3_psnr = 10 * np.log10(1.0 / grace_3_mse)
grace_3_psnr.index = (grace_3_psnr.index * 100).round()
grace_3_ssim = grace_3.groupby('loss')['ssim'].mean()
grace_3_ssim.index = (grace_3_ssim.index * 100).round()

grace_5_mse = grace_5.groupby('loss')['mse'].mean()
grace_5_psnr = 10 * np.log10(1.0 / grace_5_mse)
grace_5_psnr.index = (grace_5_psnr.index * 100).round()
grace_5_ssim = grace_5.groupby('loss')['ssim'].mean()
grace_5_ssim.index = (grace_5_ssim.index * 100).round()

# Set plot style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# NOTE: 1 consecutive frame
# MSE plot
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["MSE"], marker='o', label="PNC")
# plt.plot(castr_1["loss"], castr_1["mse"], marker='o', label="CASTR")
# plt.plot(grace_1_mse.index, grace_1_mse.values, marker='o', label="GRACE")
# plt.title("Average MSE vs. Packet Loss")
# plt.xlabel("Loss (%)")
# plt.ylabel("MSE (RGB pixel-wise)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("mse_vs_loss_1_consecutive_frame.pdf", format="pdf")  # Save as SVG
# plt.show()

# PSNR plot
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker='o', label="PNC")
plt.plot(castr_1["loss"], castr_1["psnr"], marker='o', label="CASTR")
plt.plot(grace_1_psnr.index, grace_1_psnr.values, marker='o', label="GRACE")
plt.title("Average PSNR vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("psnr_vs_loss_1_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()

# SSIM plot
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker='o', label="PNC")
plt.plot(castr_1["loss"], castr_1["ssim"], marker='o', label="CASTR")
plt.plot(grace_1_ssim.index, grace_1_ssim.values, marker='o', label="GRACE")
plt.title("Average SSIM vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("SSIM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ssim_vs_loss_1_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()


# NOTE: 3 consecutive frames
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker='o', label="PNC")
plt.plot(castr_3["loss"], castr_3["psnr"], marker='o', label="CASTR")
plt.plot(grace_3_psnr.index, grace_3_psnr.values, marker='o', label="GRACE")
plt.title("Average PSNR vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("psnr_vs_loss_3_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()

# SSIM plot
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker='o', label="PNC")
plt.plot(castr_3["loss"], castr_3["ssim"], marker='o', label="CASTR")
plt.plot(grace_3_ssim.index, grace_3_ssim.values, marker='o', label="GRACE")
plt.title("Average SSIM vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("SSIM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ssim_vs_loss_3_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()


# NOTE: 5 consecutive frames
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker='o', label="PNC")
plt.plot(castr_5["loss"], castr_5["psnr"], marker='o', label="CASTR")
plt.plot(grace_5_psnr.index, grace_5_psnr.values, marker='o', label="GRACE")
plt.title("Average PSNR vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("psnr_vs_loss_5_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()

# SSIM plot
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker='o', label="PNC")
plt.plot(castr_5["loss"], castr_5["ssim"], marker='o', label="CASTR")
plt.plot(grace_5_ssim.index, grace_5_ssim.values, marker='o', label="GRACE")
plt.title("Average SSIM vs. % Loss")
plt.xlabel("Loss (%)")
plt.ylabel("SSIM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ssim_vs_loss_5_consecutive_frame.pdf", format="pdf")  # Save as SVG
plt.show()




###### NOTE: uncomment the below code to get frame affect first frame ######
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker='o', label="PNC")
# plt.plot(castr_1["loss"], castr_1["psnr"], marker='o', label="CASTR")
# # plt.plot(grace_1_psnr.index, grace_1_psnr.values, marker='o', label="GRACE")
# plt.title("Average PSNR vs. % Loss (GRACE unavailable for this scenario)")
# plt.xlabel("Loss (%)")
# plt.ylabel("PSNR (dB)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("psnr_vs_loss_frame_1_affected.pdf", format="pdf")  # Save as SVG
# plt.show()

# # SSIM plot
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker='o', label="PNC")
# plt.plot(castr_1["loss"], castr_1["ssim"], marker='o', label="CASTR")
# # plt.plot(grace_1_ssim.index, grace_1_ssim.values, marker='o', label="GRACE")
# plt.title("Average SSIM vs. % Loss (GRACE unavailable for this scenario)")
# plt.xlabel("Loss (%)")
# plt.ylabel("SSIM")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("ssim_vs_loss_frame_1_affected.pdf", format="pdf")  # Save as SVG
# plt.show()
