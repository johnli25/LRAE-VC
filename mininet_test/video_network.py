from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.term import makeTerm
import time
import os

class VideoNetworkTopo(Topo):
    # Simple send/receive topology for video compression
    def build(self):
        sender = self.addHost("h1")
        receiver = self.addHost("h2")
        switch = self.addSwitch("s1")

        # link delays and drop rates could be varied later (params generally variable)
        self.addLink(sender, switch, bw=1000, delay="5ms")
        self.addLink(receiver, switch, bw=1000, delay="5ms")

def transmission_test():
    """Test the video transmission over the network."""
    setLogLevel('info')
    topo = VideoNetworkTopo()
    
    net = Mininet(topo=topo, link=TCLink, waitConnected=True)
    
    try:
        net.start()

        print("Dumping host connections:")
        dumpNodeConnections(net.hosts)

        sender = net.get("h1")
        receiver = net.get("h2")
        
        # Create necessary directories
        for host in [sender, receiver]:
            host.cmd("mkdir -p ~/LRAE")
            host.cmd("mkdir -p ~/LRAE/models")
            host.cmd("mkdir -p ~/LRAE/mininet_test")
        
        # Copy files to both hosts
        print("Copying necessary files to hosts...")
        
        # Copy Python scripts
        sender.cmd("cp mininet_sender.py ~/LRAE/mininet_test/")
        receiver.cmd("cp mininet_receiver.py ~/LRAE/mininet_test/")
        
        # Copy models.py
        for host in [sender, receiver]:
            host.cmd("cp ../models.py ~/LRAE/")
        
        # Copy all of the model weights over 
        for host in [sender, receiver]:
            host.cmd("cp ../conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth ~/LRAE/")
            host.cmd("cp ../PNC32_final_w_taildrops.pth ~/LRAE/")
            host.cmd("cp ../conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth ~/LRAE/")
        
        # For sender only, copy video files
        print("Copying video files to sender (this might take a while)...")
        sender.cmd("cp -r ../TUCF_sports_action_224x224_mp4_vids ~/LRAE/")
        
        # Copy the shell scripts
        print("Copying scripts to hosts...")
        sender.cmd("cp sender.sh ~/")
        receiver.cmd("cp receiver.sh ~/")
        
        # Make them executable
        sender.cmd("chmod +x ~/sender.sh")
        receiver.cmd("chmod +x ~/receiver.sh")
        
        # Launch receiver in its own terminal
        print("Starting receiver in new terminal...")
        receiver_cmd = "cd ~ && ./receiver.sh"
        makeTerm(receiver, cmd=f"bash -c \"{receiver_cmd}; read -p 'Press Enter to close...'\"")

        time.sleep(2)  

        # Launch sender in its own terminal
        print("Starting sender in new terminal...")
        sender_cmd = "cd ~ && ./sender.sh"
        makeTerm(sender, cmd=f"bash -c \"{sender_cmd}; read -p 'Press Enter to close...'\"")
        
        # Keep the network running until user interrupts
        print("\nNetwork is running. Press Ctrl+C to stop...")
        print("When you're done, the receiver_output directory will be copied to your host machine.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Create a directory on the host machine for the results
            host_output_dir = "receiver_results"
            if not os.path.exists(host_output_dir):
                os.makedirs(host_output_dir)
            
        # Copy receiver output directly to host machine
        print("Copying receiver output to host machine...")
        # First make sure the directory exists on the receiver
        receiver_files = receiver.cmd("ls -la /root/receiver_frames/ | wc -l").strip()
        print(f"Found {receiver_files} entries in receiver's output directory")
        
        if receiver.cmd("ls -la /root/ | grep receiver_frames").strip():
            # Create a temporary directory in /tmp for copying
            os.system("mkdir -p /tmp/receiver_frames_temp")
            
            # Get a list of files from the receiver
            file_list = receiver.cmd("ls -1 /root/receiver_frames/").splitlines()
            print(f"Found {len(file_list)} files in receiver_frames directory")
            
            # Copy each file individually and verify
            for filename in file_list:
                filename = filename.strip()
                if not filename:  # Skip empty lines
                    continue
                
                print(f"Copying file: {filename}")
                receiver.cmd(f"cp /root/receiver_frames/{filename} /tmp/receiver_frames_temp/")
                
                # Verify the file was copied to /tmp
                if not os.path.exists(f"/tmp/receiver_frames_temp/{filename}"):
                    print(f"Warning: Failed to copy {filename} to temp directory")
            
            # Count files in temp directory
            temp_count = len([f for f in os.listdir("/tmp/receiver_frames_temp/") if os.path.isfile(f"/tmp/receiver_frames_temp/{f}")])
            print(f"Number of files in temp directory: {temp_count}")
            
            # Copy from /tmp to host_output_dir with verbose flag
            os.system(f"cp -v /tmp/receiver_frames_temp/* {host_output_dir}/")
            
            # Verify copy success
            host_count = len([f for f in os.listdir(host_output_dir) if os.path.isfile(f"{host_output_dir}/{f}")])
            print(f"Number of files in host directory: {host_count}")
            
            # Cleanup
            os.system("rm -rf /tmp/receiver_frames_temp")
            
            if host_count != len(file_list):
                print(f"Warning: Not all files were copied. Expected {len(file_list)}, got {host_count}")
            else:
                print(f"All {host_count} files successfully copied to {host_output_dir}/")
        else:
            print("No receiver_frames directory found on receiver host")
            
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        net.stop()
        print("Network stopped.")
if __name__ == "__main__":
    # Must be run as root/sudo
    if os.geteuid() != 0:
        print("Please run as root/with sudo")
        exit(1)
    transmission_test()