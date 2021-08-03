import pathlib
import subprocess
import os
import argparse
import paramiko
from argparse import Namespace

servers = {
    'Linux2': {'name': 'Linux2a', 'pass': '1234', 'user': 'tzahi'},
    'Linux3': {'name': 'Linux3', 'pass': 'tzahi123', 'user': 'tzahi'},
    'Linux4': {'name': 'Linux4', 'pass': 'tz', 'user': 'tzahi'},
    'Lee': {'name': 'naama-server1.ef.technion.ac.il', 'pass': 'tzahi123', 'user': 'tzahi'}
}
project_paths = {
    'hucrl': '/home/tzahi/workarea/hucrl',
    'TD3-master': '/home/tzahi/workarea/TD3_master'
}


def scp_copy(args):
    for server in args.servers:

        # connect to client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, username=servers[server]['user'], password=servers[server]['pass'])

        # locate all files with the posfix
        project_path = project_paths[args.project]
        cmd = f"find {project_path}/{args.src_path} -name '*{args.files_postfix}'"
        stdin, stdout, stderr = client.exec_command(cmd)

        sftp = client.open_sftp()
        for line in stdout:
            remotepath = line.strip('\n')
            # if not ('BPTT' in remotepath and 'MBInv' in remotepath and 'Aug' in remotepath):
            #     continue

            localpath = f"{args.dst_path}/{remotepath.replace(f'{project_path}/{args.src_path}/', '')}"

            localpath = localpath.replace('/', '\\')
            if not os.path.exists(localpath):
                os.makedirs(os.path.dirname(localpath), exist_ok=True)
                sftp.get(remotepath, localpath)

        sftp.close()
        client.close()
        del client


def get_defualt_hucrl_args():
    args = {
        "project": 'hucrl',
        "servers": ['Linux3', "Linux4"],
        "src_path": 'runs',
        "dst_path": 'runs',
        "files_postfix": '.json',
    }
    args = Namespace(**args)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for H-UCRL.")
    parser.add_argument("--project", type=str, default='hucrl')
    parser.add_argument("--servers", metavar='N', nargs='+', type=str, default=['Linux3', "Linux4"])
    parser.add_argument("--src-path", type=str, default='runs')
    parser.add_argument("--dst-path", type=str, default='runs')
    parser.add_argument('--files-postfix', type=str, default='.json')
    args = parser.parse_args()

    scp_copy(args)
