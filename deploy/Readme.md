##Dockerfile Info

Install fabric through pip to be able to use the deployment script (fabfile.py).
`pip install fab`

The following should run on a local terminal, make sure your current directory is ibbm-main/caffe-docker/
because this is where the fabfile.py is.

Command to setup the docker containers for the first time :
fab setup_deepliver -A -H username@HOST -p <password of username>

HOST is the IP of the machine to deploy on.

Command to run the container and verify GPU is working (it will run nvidia-smi) :
fab run_deepliver -A -H username@HOST -p <password for username>

Example : fab setup_container -A -H mohamedezz@10.162.254.76 -p mysecretpassword

Note : The option -A passes over your ssh identity (private key), so that you can pull our private repos from bitbucket. This requires that you register your private key to be used by the ssh forwarding agent. Do this :

`ssh-add /path/to/private_key`

e.g. : `ssh-add ~/.ssh/id_rsa`

Note for first-time-usage: Make sure that '~/.ssh/known_hosts' exists on the HOST machine. Otherwise you might get a runtime error.

##Add User to Docker Group
You need to add your current user to docker group as follow :
`sudo usermod -aG docker <user>`
then logout & login again into the system or restart the system. test by:
`docker run hello-world`


##Docker Containers Usage
To easily use the docker containers on the machine they're deployed on, you can add the following aliases in your ~/.bashrc
for convenience :
`alias trainhere='sudo GPU=0 nvidia-docker run -v $(pwd):/data -w /data -ti mohamedezz/jonlong bash -c "caffe train -solver solver_deepliver.prototxt" && sudo docker rm $(sudo docker ps -l -q)'

`alias caffecont='sudo GPU=0 nvidia-docker run -v /home/mohamedezz:/data -p 8888:8888 --net=host --privileged -w /data -ti mohamedezz/jonlong bash'`

`alias jupytercont='sudo GPU=0 nvidia-docker run -v /home/mohamedezz:/data -p 8888:8888 --net=host --privileged -w /data mohamedezz/jonlong bash -c "jupyter notebook"'`

`trainhere` : To use it, first cd into a directory which contains a file solver_deepliver.prototxt and the corresponding  network prototxts and a 2 leveldbs named train_img and train_seg (images and their segmentations).

`caffecontainer` : Use this to create a temporary caffe container.  

Both alias commands will mount the current directory from which you run the alias, to /data in the container. Note that those aliases will delete the created container after they exit, so it's a good practice to put any output data in the mounted directory /data, because any data outside this mount is gone when the container is deleted.

### Lasagne Docker ###
Can be pulled from Dockerhub
`docker pull fletling/lasagne`

or be built with 
`docker build -t fletling/lasagne .`
from the directory of the Dockerfile

Then use it similar to caffecontainer and add alias for simplicity
`alias lasagnecont='sudo GPU=0 nvidia-docker run -v /home/username:/data -p 8888:8888 --net=host --privileged -w /data -ti fletling/lasagne bash'`



