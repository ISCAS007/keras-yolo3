wget https://mirrors.ustc.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
echo "export PATH=\$HOME/miniconda3/bin:\$PATH" >> ~/.zshrc
echo "export PATH=\$HOME/miniconda3/bin:\$PATH" >> ~/.bashrc
export PATH=$HOME/miniconda3/bin:$PATH
conda create -n env3 python=3
source activate env3
conda install tensorflow-gpu=1.6 Pillow matplotlib ipython keras=2.1.5
pip install opencv_python
