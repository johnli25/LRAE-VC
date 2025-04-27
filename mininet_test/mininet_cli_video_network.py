from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
import time
import os

class VideoNetworkTopo(Topo):
    # Simple send/receive topology for video compression
    def build(self):
        sender = self.addHost("h1")
        receiver = self.addHost("h2")
        switch = self.addSwitch("s1")

        self.addLink(sender, switch, bw=1000, delay="5ms")
        self.addLink(receiver, switch, bw=1000, delay="5ms")

def setup_mininet():
    """Set up the Mininet network and prepare the hosts."""
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
        
        # Copy files to hosts
        print("Copying necessary files to hosts...")

        sender.cmd("cp mininet_sender.py ~/LRAE/mininet_test/")
        receiver.cmd("cp mininet_receiver.py ~/LRAE/mininet_test/")
        
        for host in [sender, receiver]:
            host.cmd("cp ../models.py ~/LRAE/")
            host.cmd("cp ../conv_lstm_PNC32_ae_dropUpTo_32_final_pristine.pth ~/LRAE/")
            host.cmd("cp ../PNC32_final_w_taildrops.pth ~/LRAE/")
            host.cmd("cp ../conv_lstm_PNC32_ae_dropUpTo_32_bidirectional_final.pth ~/LRAE/")
        
        print("Copying video files to sender...")
        sender.cmd("cp -r ../TUCF_sports_action_224x224_mp4_vids ~/LRAE/")
        
        print("Copying scripts to hosts...")
        sender.cmd("cp sender.sh ~/")
        receiver.cmd("cp receiver.sh ~/")
        
        sender.cmd("chmod +x ~/sender.sh")
        receiver.cmd("chmod +x ~/receiver.sh")
        
        # DROP INTO CLI
        print("\nSetup complete. Dropping into Mininet CLI.")
        print("Manually run:")
        print("  h2 ./receiver.sh")
        print("  h1 ./sender.sh")
        print("\nType 'exit' when done.\n")
        
        CLI(net)

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
    setup_mininet()
