import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSVs
pnc_df = pd.read_csv("PNC_results.csv")
castr_df = pd.read_csv("CASTR_results_loss_affecting_frame1.csv")
# castr_df = pd.read_csv("CASTR_results.csv")
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
    (grace_df['model_id'].isin([256, 512, 1024])) &
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
     ])) &
    (grace_df['frame_id'] != 0)
]

grace_1 = grace_df_base[grace_df_base['nframes'] == 1]
grace_3 = grace_df_base[grace_df_base['nframes'] == 3]
grace_5 = grace_df_base[grace_df_base['nframes'] == 5]

# Set plot style
plt.rcParams.update({
    'font.size': 36,
    'axes.titlesize': 36,
    'axes.labelsize': 36,
    'xtick.labelsize': 36,
    'ytick.labelsize': 36,
    'legend.fontsize': 40,
    'figure.titlesize': 20
})

def set_y_axis_limits(plot_type):
    pass
    # if plot_type == "PSNR":
    #     plt.ylim(15, 33)  # Adjust PSNR y-axis range
    # elif plot_type == "SSIM":
    #     plt.ylim(0.5, 1.2)  # Adjust SSIM y-axis range

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

####################################
# 1 consecutive frame plots
####################################
# alpha_label_map = {
#     256: r"$\alpha = 2^{-8}$",
#     512: r"$\alpha = 2^{-9}$",
#     1024: r"$\alpha = 2^{-10}$"
# }

# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker="o", label="PNC")
# plt.plot(castr_1["loss"], castr_1["psnr"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_1[grace_1["model_id"] == mid]
#     grace_mse = grace_subset.groupby("loss")["mse"].mean()
#     grace_psnr = 10 * np.log10(1.0 / grace_mse)
#     grace_psnr.index = (grace_psnr.index * 100).round()

#     plt.plot(
#         grace_psnr.index,
#         grace_psnr.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average PSNR vs. Packet Loss (1 consecutive frame)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("PSNR (dB)")
# set_y_axis_limits("PSNR")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("psnr_vs_loss_1_consecutive_frame.pdf", format="pdf")
# plt.show()

# # SSIM plot (1 consecutive frame)
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker="o", label="PNC")
# plt.plot(castr_1["loss"], castr_1["ssim"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_1[grace_1["model_id"] == mid]
#     grace_ssim = grace_subset.groupby("loss")["ssim"].mean()
#     grace_ssim.index = (grace_ssim.index * 100).round()

#     plt.plot(
#         grace_ssim.index,
#         grace_ssim.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average SSIM vs. Packet Loss (1 consecutive frame)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("SSIM")
# set_y_axis_limits("SSIM")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("ssim_vs_loss_1_consecutive_frame.pdf", format="pdf")
# plt.show()


# # NOTE: 3 consecutive frames
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker="o", label="PNC")
# plt.plot(castr_3["loss"], castr_3["psnr"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_3[grace_3["model_id"] == mid]
#     grace_mse = grace_subset.groupby("loss")["mse"].mean()
#     grace_psnr = 10 * np.log10(1.0 / grace_mse)
#     grace_psnr.index = (grace_psnr.index * 100).round()

#     plt.plot(
#         grace_psnr.index,
#         grace_psnr.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average PSNR vs. Packet Loss (3 consecutive frames)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("PSNR (dB)")
# set_y_axis_limits("PSNR")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("psnr_vs_loss_3_consecutive_frame.pdf", format="pdf")
# plt.show()

# # SSIM plot (3 consecutive frames)
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker="o", label="PNC")
# plt.plot(castr_3["loss"], castr_3["ssim"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_3[grace_3["model_id"] == mid]
#     grace_ssim = grace_subset.groupby("loss")["ssim"].mean()
#     grace_ssim.index = (grace_ssim.index * 100).round()

#     plt.plot(
#         grace_ssim.index,
#         grace_ssim.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average SSIM vs. Packet Loss (3 consecutive frames)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("SSIM")
# set_y_axis_limits("SSIM")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("ssim_vs_loss_3_consecutive_frame.pdf", format="pdf")
# plt.show()


# # NOTE: 5 consecutive frames
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker="o", label="PNC")
# plt.plot(castr_5["loss"], castr_5["psnr"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_5[grace_5["model_id"] == mid]
#     grace_mse = grace_subset.groupby("loss")["mse"].mean()
#     grace_psnr = 10 * np.log10(1.0 / grace_mse)
#     grace_psnr.index = (grace_psnr.index * 100).round()

#     plt.plot(
#         grace_psnr.index,
#         grace_psnr.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average PSNR vs. Packet Loss (5 consecutive frames)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("PSNR (dB)")
# set_y_axis_limits("PSNR")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("psnr_vs_loss_5_consecutive_frame.pdf", format="pdf")
# plt.show()

# # SSIM plot (5 consecutive frames)
# plt.figure(figsize=(10, 6))
# plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker="o", label="PNC")
# plt.plot(castr_5["loss"], castr_5["ssim"], marker="o", label="CASTR")

# for mid in [256, 512, 1024]:
#     grace_subset = grace_5[grace_5["model_id"] == mid]
#     grace_ssim = grace_subset.groupby("loss")["ssim"].mean()
#     grace_ssim.index = (grace_ssim.index * 100).round()

#     plt.plot(
#         grace_ssim.index,
#         grace_ssim.values,
#         marker="o",
#         label=f"GRACE ({alpha_label_map[mid]})",
#     )

# # plt.title("Average SSIM vs. Packet Loss (5 consecutive frames)")
# plt.xlabel("Packet Loss (%)")
# plt.ylabel("SSIM")
# set_y_axis_limits("SSIM")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("ssim_vs_loss_5_consecutive_frame.pdf", format="pdf")
# plt.show()




###### NOTE: uncomment the below code to get the scenario where loss affects first frame ######
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["PSNR"], marker='o', label="PNC")
plt.plot(castr_1["loss"], castr_1["psnr"], marker='o', label="CASTR")
# plt.title("Average PSNR vs. Packet Loss")
plt.xlabel("Packet Loss (%)")
plt.ylabel("PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("psnr_vs_loss_frame_1_affected.pdf", format="pdf")  # Save as SVG
plt.show()

# SSIM plot
plt.figure(figsize=(10, 6))
plt.plot(pnc_df["loss"], pnc_df["SSIM"], marker='o', label="PNC")
plt.plot(castr_1["loss"], castr_1["ssim"], marker='o', label="CASTR")
# plt.title("Average SSIM vs. Packet Loss")
plt.xlabel("Packet Loss (%)")
plt.ylabel("SSIM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ssim_vs_loss_frame_1_affected.pdf", format="pdf")  # Save as SVG
plt.show()
