import torch
import torch.nn as nn
import socket
import argparse
import matplotlib.pyplot as plt

class LRAE_VC_Decoder(nn.Module):
    def __init__(self):
        super(LRAE_VC_Decoder, self).__init__()
        self.decoder1 = nn.ConvTranspose2d(10, 64, kernel_size=9, stride=7, padding=4, output_padding=6)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.decoder3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y1 = self.relu(self.decoder1(x))
        y2 = self.relu(self.decoder2(y1))
        y2 = y2 + y1
        y3 = self.relu(self.decoder3(y2))
        y4 = self.relu(self.decoder3(y3))
        y4 = y4 + y3
        y5 = self.final_layer(y4)
        y5 = torch.clamp(y5, min=0, max=1)
        return y5


def decode_and_classify(model, list_of_enc_imgs):

    received_decoded_directory = "received_decoded_images"
    with torch.no_grad():
        for i, enc_img in enumerate(list_of_enc_imgs):
            # encoded_img = torch.tensor(enc_img).float()
            decoded_img, label = model(enc_img)
            # Save the decoded image
            decoded_img = decoded_img.squeeze(0).permute(1, 2, 0).numpy()
            plt.imsave(f"{received_decoded_directory}/decoded_img_{i}.png", decoded_img)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode images using LRAE_VC_Decoder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--ip', type=str, required=True, help='IP address to bind the server')
    parser.add_argument('--port', type=int, required=True, help='Port number to bind the server')
    args = parser.parse_args()

    model = LRAE_VC_Decoder()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((args.ip, args.port))
    server_socket.listen(1)
    conn, addr = server_socket.accept()



    conn.close()