# Parallel_DNN_HPC
This research investigates the optimization of Deep Neural Networks (DNNs) training through a comparative analysis of two parallelization methods—Allreduce and Voting—implemented via the Message Passing Interface (MPI) on CIFAR10 and CIFAR100 datasets. 

# HPC environment:
# install modules 
based on AUHPCS policies: Users cannot install environment modules into the central cluster repository, but they can in their home directories or shared lab directories.
AUHPCS Anaconda, conda path-based environments
The anaconda3 instance available on the AUHPCS cluster will provide the base anaconda software used to create conda-managed environments. Once loaded it can be used used to install/extend additional Python applications in home directories.
This conserves space by not having to host the Anaconda application itself in your AUHPCS home directory.
The AUHPCS recommendation is to create "path-based" environments in your home directory to install specific Python applications/environments to. Although there may be package redundancy using discreet paths for Python application environments, these environments may be more intuitive to manage when compared to "name-based" environments.

### Load the Anaconda module from the cluster:
~# module load anaconda3/2021.11
Run the conda initialize command to configure your parent conda environment:
~# conda init
- This will configure your ~/.bashrc to reference the /cm/shared/apps/Anaconda3/2021.11/ path for all future conda commands.
- For the changes to take effect, close and restart your login shell.

Following conda initialization, conda, and python applications can be installed by activating the target environment and using either the "conda install" or pip/pip3 commands.
Create a path for the new application environment:
~# mkdir ~/apps/conda/appx
Create and activate the path-based environment:
~# conda create -p ~/apps/conda/appx
~# conda activate ~/apps/conda/appx
Then install your conda application:
~# conda install appx
Or, if a specific software channel is required:
~# conda install -c conda-forge appx
### You can then activate your environment in the manner already detailed to use the installed software
### Using Anaconda-managed applications in interactive computing sessions
~# module load anaconda3/2021.11
~# conda activate ~/apps/conda/appx
### Using Anaconda-managed applications in SLURM job submission scripts
~# module load anaconda3/2021.11
~# source activate ~/apps/conda/appx

### Running Python applications from conda environments (apply the above conventions to interactive and batch job submissions)
~# module load anaconda3/2021.11
~# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
~# conda activate myenv
~# srun python /<FULL_PATH>/myscript.py

 
