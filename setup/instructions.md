ok so here is how to setup the env in the cluster:

first mkdir -p /scratch/gmuid/waterdiffusion
cd /scratch/gmuid/waterdiffusion
module load gnu10
module load python/3.9.9-jh
which python
- use the output of which python for the below file path
/opt/sw/spack/apps/linux-rhel8-x86_64_v2/gcc-10.3.0/python-3.9.9-jh/bin/python3.9 -m venv /scratch/gmuid/waterdiffusion/watervenv

source watervenv/bin/activate

setup github:

ssh-keygen -t ed25519 -C "cluster_name_github_key - your_email@example.com"
cat ~/.ssh/id_ed25519.pub
copy this output to your github account in ssh keys
clone the repo


pip install -r waterdiffusion/setup/requirements.txt

ok yay done now copy and paste the submit_job.sh
to run this do sbatch submit_job.sh
nano submit_job.sh to make the file