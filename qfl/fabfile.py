import boto, urllib2
from   boto.ec2 import connect_to_region
from   fabric.api import env, run, cd, settings, sudo
from   fabric.api import parallel
import os
import sys

REGION = os.environ.get("AWS_EC2_REGION")
WEB_ROOT = "/var/www"

# Server user, normally AWS Ubuntu instances have default user "ubuntu"
env.user = "ec2-user"

# List of AWS private key Files
env.key_filename = ["~/.ssh/ec2-sample-key.pem"]


def hello(name="world"):
    """
    Test function
    """
    print("Hello %s!" % name)


# Fabric task to restart Apache, runs in parallel
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 restart_apache
@parallel
def reload_apache():
    sudo('service apache restart')


# Fabric task to start Apache, runs in parallel
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 start_apache
@parallel
def start_apache():
    sudo('service apache start')


# Fabric task to stop Apache, runs in parallel
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 stop_apache
@parallel
def stop_apache():
    sudo('service apache stop')


# Fabric task to updates/upgrade OS (Ubuntu), runs in parallel
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 update_os
@parallel
def update_os():
    sudo('apt-get update -y')
    sudo('apt-get upgrade -y')


# Fabric task to reboot OS (Ubuntu), runs in parallel
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 reboot_os
@parallel
def reboot_os():
    sudo('reboot')


# Fabric task for cloning GIT repository in Apache WEB_ROOT
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 update_branch restart_apache
@parallel
def clone_branch():
    with cd("/var/www"):
        run('git clone https://www.github.com/user/repo.git')


# Fabric task for deploying latest changes using GIT pull
# This assumes that your GIT repository is in Apache WEB_ROOT
# To execute task using fabric run following
# fab set_hosts:phpapp,2X,us-west-1 update_branch restart_apache
@parallel
def update_branch():
    with cd("/var/www"):
        run('git pull -f')

# Your custom Fabric task here after and run them using,
# fab set_hosts:phpapp,2X,us-west-1 task1 task2 task3


# Fabric task to set env.hosts based on tag key-value pair
def set_hosts(tag = "phpapp", value="*", region=REGION):
    key = "tag:"+tag
    env.hosts = _get_public_dns(region, key, value)


# Private method to get public DNS name for instance with
#  given tag key and value pair
def _get_public_dns(region, key, value ="*"):
    public_dns = []
    connection = _create_connection(region)
    reservations = connection.get_all_instances(filters = {key : value})
    for reservation in reservations:
        for instance in reservation.instances:
            print "Instance", instance.public_dns_name
            public_dns.append(str(instance.public_dns_name))
    return public_dns


# Private method for getting AWS connection
def _create_connection(region):
    print "Connecting to ", region

    conn = connect_to_region(
        region_name = region
    )

    print "Connection with AWS established"
    return conn