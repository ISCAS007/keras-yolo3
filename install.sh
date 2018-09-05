wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.zshrc
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.bashrc
export PATH=$HOME/miniconda/bin:$PATH
conda create -n env3 python=3
source activate env3
conda install tensorflow-gpu=1.6 Pillow matlabplot
pip install opencv_python
