import boto
from boto.s3.key import Key
from boto.s3.connection import OrdinaryCallingFormat
import boto.ec2
import simplejson, uuid, time, sys
from fabric.api import env, run, settings, sudo
from ilogue.fexpect import expect, expecting, run
import os

aws_region = 'us-west-2'
ec2_key_name = 'ec2-sample-key'
ami = 'ami-9fac6cff'
ssh_loc = '/Users/benneifert/.ssh'
security_group_name = 'appserver'
cidr_ip = '192.168.1.66/32'
port_num = 22
instance_type = 't2.micro'

MAIN_IP = '52-42-24-21'
MAIN_SG = 'appserver'
MAIN_KP = 'ec2-sample-key'

SERVER_SETTINGS = {
    'app': {
           'image_id': ami,
           'instance_type': instance_type,
           'security_groups': [MAIN_SG],
           'key_name' : MAIN_KP,
            },
}


class EC2Conn:

        def __init__(self):
                self.conn = None

        def connect(self):
                # self.conn = boto.ec2.EC2Connection()
                self.conn = boto.ec2.connect_to_region(aws_region)

        def create_instance(self, server_type='app', address=None):

                reservation = self.conn.run_instances(
                    **SERVER_SETTINGS[server_type])

                # reservation = self.conn.run_instances(
                #     image_id=SERVER_SETTINGS[server_type]['image_ami'],
                #     key_name=SERVER_SETTINGS[server_type]['key_name'],
                #     security_group_ids=SERVER_SETTINGS[server_type]['security_groups'],
                #     instance_type=SERVER_SETTINGS[server_type]['instance_type'])

                print reservation

                instance = reservation.instances[0]
                time.sleep(10)
                while instance.state != 'running':
                        time.sleep(5)
                        instance.update()
                        print "Instance state: %s" % (instance.state)

                print "instance %s done!" % (instance.id)

                if address:
                        success = self.link_instance_and_ip(instance.id, address)
                        if success:
                                print "Linked %s to %s" % (instance.id, address)
                        else:
                                print "Failed to link%s to %s" % (instance.id, address)
                        instance.update()

                return instance

        def link_instance_and_ip(self, instance_id, ip=MAIN_IP):
                success = self.conn.associate_address(instance_id=instance_id,
                                                      public_ip=ip)
                if success:
                        print "Sleeping for 60 seconds to let IP attach"
                        time.sleep(60)

                return success

        def unlink_instance_and_ip(self, instance_id, ip=MAIN_IP):
                return self.conn.disassociate_address(instance_id=instance_id,
                                                      public_ip=ip)

        def get_instances(self):
                return self.conn.get_all_instances()


def create_new_instance(address=MAIN_IP):
        a = EC2Conn()
        a.connect()
        return a.create_instance(address=address)


def main():

################################################################################
# S3
################################################################################

        s3 = boto.connect_s3(
            is_secure=True,
            calling_format=OrdinaryCallingFormat(),
        )

        # Creating a bucket
        # bucket = s3.create_bucket('benns-new-bucket')

        # Getting a bucket
        bucket = s3.get_bucket('benns-new-bucket')

        # Put something in a bucket
        k = Key(bucket)
        k.key = 'item_1'
        k.set_contents_from_string('This is a test of S3')

        # Get something from bucket
        k = Key(bucket)
        k.key = 'item_1'
        k.get_contents_as_string()

        # Check what keys are in the bucket
        bucket.get_all_keys()


################################################################################
# SQS
################################################################################

        # Connect to SQS
        sqs = boto.connect_sqs()

        # Connect to S3
        s3 = boto.connect_s3()
        bucket = s3.get_bucket('benns-new-bucket')

        # Create a queue
        q = sqs.create_queue('my_message_pump')

        # Create a new message
        data = simplejson.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
        key = bucket.new_key('2010-03-20/%s.json' % str(uuid.uuid4()))
        key.set_contents_from_string(data)
        message = q.new_message(body=simplejson.dumps(
            {'bucket': bucket.name, 'key': key.name}))
        q.write(message)

        # Read the message
        q = sqs.get_queue('my_message_pump')
        message = q.read()
        if message is not None:
            msg_data = simplejson.loads(message.get_body())
            key = boto.connect_s3().get_bucket(
                msg_data['bucket']).get_key(msg_data['key'])
            data = simplejson.loads(key.get_contents_as_string())
            q.delete_message(message)


################################################################################
# EC2
################################################################################

        # # Configure new stuff
        # # only needs to be done once
        # key_pair = ec2.create_key_pair(ec2_key_name)
        # key_pair.save(ssh_loc)

        ## Setup new authorization
        # app.authorize(ip_protocol='tcp',
        #               from_port=port_num,
        #               to_port=port_num,
        #               cidr_ip=cidr_ip)

        ec2 = boto.ec2.connect_to_region(aws_region)
        # us-west-2 specific AMI

        # us-west-1 specific AMI
        # ami = 'ami-a2490dc2'


        # reservation = ec2.run_instances(image_id=ami,
        #                                 key_name=ec2_key_name,
        #                                 security_group_ids=[security_group_name])

        r = ec2.get_all_instances()[2]
        instance = r.instances[0]
        instance.ip_address

        # app = ec2.create_security_group(security_group_name, 'Application tier')
        app = ec2.get_all_security_groups(groupnames=[security_group_name])[0]

        # Create New Instance
        # instance = create_new_instance()

        # Set Env Host String
        env.host_string = "root-user@%s" % (instance.ip_address)
        env['key_filename'] = '~/.ssh/ec2-sample-key.pem'
        env['user'] = 'root'

        # env['hosts'] = ['ec2-52-42-53-34.us-west-2.compute.amazonaws.com']

        env['password'] = 'Thirdgen1'
        env['localuser'] = 'benneifert'
        env['use_shell'] = False
        env['sudo_user'] = 'root'
        env['user'] = 'root'

        env.prompts = {'Is this ok [y/d/N]': 'y',
                       'Is this ok [y/d/N]:': 'y',
                       'Is this ok [y/d/N]: ': 'y'}

        sudo('yum update')

        # stop instance
        # ec2.stop_instances([instance.id])


        # Begin Installation

        user = 'benneifert'
        remote_home_dir = '/home/' + user

        run('echo "{0} ALL=(ALL) ALL" >> /etc/sudoers'.format(env.user))

        with settings(warn_only=True):
            sudo('sh', shell=False)
            sudo('useradd -U -m %s, shell=false' % user)

        # Install packages with yum
        sudo('yum install -y %s' % (" ".join(PACKAGES_LIST)))

        # Install pip
        sudo('curl -O http://pypi.python.org/packages/source/p/pip/pip-1.0.tar.gz')
        run('tar xvfz pip-1.0.tar.gz')
        sudo('cd pip-1.0 && python setup.py install')

        # Install virtualenv
        sudo('pip install virtualenv')
        venv_name = '%s-env' % user
        venv = os.path.join(remote_home_dir, venv_name)
        sudo('virtualenv --no-site-packages %s' % venv)

        # Install python requirements
        # put('requirements.txt', remote_home_dir, use_sudo=True)
        sudo('%s/bin/pip install -r %s/requirements.txt' % (venv, remote_home_dir))

        # deploy_the_codes(HOME_DIR, remote_home_dir)


