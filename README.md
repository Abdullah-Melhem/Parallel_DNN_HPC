# Parallel_DNN_HPC
This research investigates the optimization of Deep Neural Networks (DNNs) training through a comparative analysis of two parallelization methods—Allreduce and Voting—implemented via the Message Passing Interface (MPI) on CIFAR10 and CIFAR100 datasets. 

# HPC environment:
# install modules 
-- Based on AUHPCS policies: Users cannot install environment modules into the central cluster repository, but they can in their home directories or shared lab directories.
AUHPCS Anaconda, conda path-based environments
-- The anaconda3 instance available on the AUHPCS cluster will provide the base anaconda software used to create conda-managed environments. Once loaded it can be used to install/extend additional Python applications in home directories.
-- This conserves space by not having to host the Anaconda application itself in your AUHPCS home directory.
-- The AUHPCS recommendation is to create "path-based" environments in your home directory to install specific Python applications/environments. Although there may be package redundancy using discreet paths for Python application environments, these environments may be more intuitive to manage when compared to "name-based" environments.

### Load the Anaconda module from the cluster:
- ~# module load anaconda3/2021.11
- Run the conda initialize command to configure your parent conda environment:
- ~# conda init
- This will configure your ~/.bashrc to reference the /cm/shared/apps/Anaconda3/2021.11/ path for all future conda commands.
- For the changes to take effect, close and restart your login shell.

- Following conda initialization, conda, and python applications can be installed by activating the target environment and using either the "conda install" or pip/pip3 commands.
- Create a path for the new application environment:
- ~# mkdir ~/apps/conda/appx
- Create and activate the path-based environment:
- ~# conda create -p ~/apps/conda/appx
- ~# conda activate ~/apps/conda/appx
- Then install your conda application:
- ~# conda install appx Or, if a specific software channel is required:
- ~# conda install -c conda-forge appx
### You can then activate your environment in the manner already detailed to use the installed software
### Using Anaconda-managed applications in interactive computing sessions
- ~# module load anaconda3/2021.11
- ~# conda activate ~/apps/conda/appx
### Using Anaconda-managed applications in SLURM job submission scripts
- ~# module load anaconda3/2021.11
- ~# source activate ~/apps/conda/appx

### Running Python applications from conda environments (apply the above conventions to interactive and batch job submissions)
- ~# module load anaconda3/2021.11
- ~# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
- ~# conda activate myenv
- ~# srun python /<FULL_PATH>/myscript.py
# SLURM Job 
Example of the SLURM file uploaded in the SLURM example.
- Open a Text Editor: Open a text editor on your system. Use a plain text editor like Notepad (on Windows), TextEdit (on macOS), or any code editor of your choice.
- Start with the Shebang Line: start with the shebang line #!/bin/bash. This line tells the system to use the Bash shell to interpret the script.
- Set the SLURM parameters using #SBATCH lines. Adjust the values according to your specific requirements.
- Load necessary modules using module load commands.
- If needed, load other modules like TensorFlow or Keras using similar module load commands.
- If there are any environment variables to set, you can do so in the script. (example: number of MPI processes).
- Run the MPI script using the mpirun command. Adjust the path and script name according to your setup.
- Save the script with a .sh extension.
 
# SLURM parameters 
- #SBATCH --job-name=Voting: Specifies the name of the job. In this case, the job is named "Voting."
- #SBATCH --nodes=1: Specifies the number of nodes requested for the job. In this case, it's requesting 1 node.
- #SBATCH --ntasks-per-node=4: Specifies the number of tasks (MPI processes) to be run per node. It's set to 4 tasks per node in this example.
- #SBATCH --cpus-per-task=4: Specifies the number of CPU cores to allocate per task. It's set to 4 CPU cores per task in this example.
- #SBATCH --time=02:10:00: Specifies the maximum amount of time the job is allowed to run. In this case, it's set to 2 hours and 10 minutes.
- #SBATCH --mail-type=NONE: Specifies the type of email notifications for the job. In this example, no email notifications will be sent (NONE).
- #SBATCH --mail-user=youremail@example.edu: Specifies the email address to which notifications will be sent. Replace youremail@example.edu with the actual email address.
- #SBATCH --output=output.out: Specifies the name of the file to which standard output (stdout) will be written. In this case, it's set to "output.out." This file will contain the output of your job.
