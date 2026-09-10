"""从 AutoDL 拉取云端训练权重（paramiko + scp，自动输密码）"""
import os
import paramiko
from scp import SCPClient

HOST = "connect.westc.seetacloud.com"
PORT = 56500
USER = "root"
PASSWORD = "Q1VeJYQ9yKZJ"

REMOTE_DIR = "/root/outputs/ckpts"
FILES = ["cloudB_v2_seed0.pt", "cloudB_v2_seed1.pt"]

LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "ckpts"))
os.makedirs(LOCAL_DIR, exist_ok=True)


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"连接 {HOST}:{PORT} ...")
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print("连接成功，列出远程权重：")
    stdin, stdout, stderr = ssh.exec_command(f"ls -lh {REMOTE_DIR}/cloud*.pt 2>&1")
    print(stdout.read().decode())

    with SCPClient(ssh.get_transport()) as scp:
        for f in FILES:
            remote = f"{REMOTE_DIR}/{f}"
            local = os.path.join(LOCAL_DIR, f)
            print(f"拉取 {f} ...", flush=True)
            scp.get(remote, local)
            size = os.path.getsize(local) / 1e6
            print(f"  完成 {f} ({size:.1f} MB)", flush=True)

    ssh.close()
    print("\n全部拉取完成！")
    print("本地文件：")
    for f in FILES:
        p = os.path.join(LOCAL_DIR, f)
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
