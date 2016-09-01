"""
Usage 1 : fab pull_container -A -H username@HOST -p <password of username>
Usage 2 : fab run_container -A -H username@HOST -p <password for username>

After docker image is built/downloaded, you can train using a command like this:
sudo GPU=0 nvidia-docker run -v $(pwd):/data mohamedezz/jonlong bash -c "cd /data && /opt/caffe/build/tools/caffe train -solver solver_fcn.prototxt"
"""

from fabric.api import *
from fabric.contrib.files import *

REPO_HOME="/opt/caffe-vnet-docker-workspace"
CONTAINER_NAME = "patrickchrist/vnet"

def failed(command, operation=run):
	"""Convenience function to check for non-zero return code of a command"""
	with settings(warn_only=True):
		result = operation(command)
	return result.failed

def init():
	sudo("apt-get update")
	sudo("apt-get install -y awscli htop git")

@task
def install_nvidia_driver(force=False):
	sudo("apt-get install -y gcc make pkg-config xorg-dev")
	if force or failed("which nvidia-smi"):
		with cd("/tmp"):
			if failed("ls NVIDIA-Linux-x86_64-352.63.run"):
				put("../NVIDIA-Linux-x86_64-352.63.run", "/tmp/")
			sudo("update-initramfs -u") #For next reboot: disable nouvaeu kernel driver
			sudo("bash NVIDIA-Linux-x86_64-352.63.run --ui=none --no-questions --accept-license --disable-nouveau")
@task
def install_docker():
	""" Install docker on a unix machine"""
	if failed("which docker"):
		sudo("apt-get install -y wget")
		sudo("wget -qO- https://get.docker.com/ | sh")

@task
def add_aliases():
	if not exists("~/.bashrc"):
		run("echo >> ~/.bashrc")
        
        #for TUM machines:	
        aliases = ["caffecontainer='sudo GPU=0 nvidia-docker run -P --net=host --volume=/media/nas:/media/nas --volume=$(pwd):/data --workdir=/data -it patrickchrist/vnet /bin/bash'"]
        #aliases = ["caffecontainer='sudo GPU=0 nvidia-docker run -P --net=host --volume=/media/hdd2:/media/hdd2 --volume=$(pwd):/data --workdir=/data -it patrickchrist/vnet /bin/bash'"]
	
	for alias in aliases:
		run('echo -e "alias '+alias+'" >> ~/.bashrc')

@task
def setup_container():
	"""Builds a docker Image ready with caffe / deepliver with GPU support
	.It will be ready to run a container from this image"""

	init()
	install_docker()
	install_nvidia_driver()
	add_aliases()
	#sudo("rm -r "+REPO_HOME)
	sudo("mkdir -p "+REPO_HOME)
	sudo("chmod 777 "+REPO_HOME)

	# Home for cloned repos
	with cd(REPO_HOME):
		# Add bitbucket to knownhosts (not necessary when using deployment ssh key without strict hostkeychecking)
		#if not exists("~/.ssh/known_hosts"):
		#	run("mkdir -p ~/.ssh")
		#	run("echo >> ~/.ssh/known_hosts")
		#run("ssh-keyscan -t rsa bitbucket.org >> ~/.ssh/known_hosts")
		#Get nvidia containers - needed for docker with GPU support
		if failed("cd "+REPO_HOME+"/nvidia-docker"):
			run('GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone https://github.com/NVIDIA/nvidia-docker.git')
			with cd("nvidia-docker"):
				run('git reset --hard dfd0188705c3044a2be0e0309cc1110ec71a19a9')

	# copy docker file
	run('mkdir -p '+REPO_HOME+"/caffe-vnet-docker")
	put('Dockerfile',REPO_HOME+"/caffe-vnet-docker/")	
        put('repo_key',REPO_HOME+"/caffe-vnet-docker/") 

	with cd(REPO_HOME+'/nvidia-docker'):
		#Build nvidia-docker layers
		sudo("docker build -t cuda:7.0-runtime ubuntu/cuda/7.0/runtime")
		sudo("docker build -t cuda:7.0-devel   ubuntu/cuda/7.0/devel")
		sudo("docker build -t cuda:7.0-cudnn3-devel ubuntu/cuda/7.0/devel/cudnn3")
		# Add nice tag
		sudo("docker tag cuda:7.0-cudnn3-devel cuda:cudnn-devel")
		# Make ./nvidia-docker command accessible anywhere
		sudo("ln -f -s "+REPO_HOME+"/nvidia-docker/nvidia-docker /usr/bin/nvidia-docker")


		
	with cd(REPO_HOME+"/caffe-vnet-docker/"):
		sudo("docker build -t vnet .")
		sudo("docker tag vnet "+CONTAINER_NAME)
	
	

@task
def pull_container():
	""" An alternative to setup_container which pulls a ready built image from dockerhub under mohamedezz/jonlong"""
	
	init()
	install_docker()
	#install_nvidia_driver()
	add_aliases()
	#sudo("rm -r "+REPO_HOME)
	sudo("mkdir -p "+REPO_HOME)
	sudo("chmod 777 "+REPO_HOME)

	# Home for cloned repos
	with cd(REPO_HOME):
		# Add bitbucket to knownhosts
		if not exists("~/.ssh/known_hosts"):
			run("mkdir -p ~/.ssh")
			run("echo >> ~/.ssh/known_hosts")
		
		run("ssh-keyscan -t rsa bitbucket.org >> ~/.ssh/known_hosts")
		#Get nvidia containers - needed for docker with GPU support
		if failed("cd "+REPO_HOME+"/nvidia-docker"):
			run('GIT_SSH_COMMAND="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" git clone https://github.com/NVIDIA/nvidia-docker.git')
			with cd("nvidia-docker"):
				run('git reset --hard dfd0188705c3044a2be0e0309cc1110ec71a19a9')
				
	sudo("ln -f -s "+REPO_HOME+"/nvidia-docker/nvidia-docker /usr/bin/nvidia-docker")
	sudo("docker pull "+ CONTAINER_NAME)
	

@task
def run_container():
	with shell_env(GPU="0"):
		sudo("nvidia-docker run "+ CONTAINER_NAME +" nvidia-smi")
		sudo("nvidia-docker run -it "+ CONTAINER_NAME +" /bin/bash")
		sudo("docker rm $(docker ps -l -q)")


