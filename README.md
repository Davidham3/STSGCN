# STSGCN
AAAI 2020. Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting

url: paper/AAAI2020-STSGCN.pdf

# Usage

Docker is recommended.

1. install docker
2. install nvidia-docker
3. build image using `cd docker && docker build -t stsgcn/mxnet_1.41_cu100 .`
4. download the data [STSGCN_data.tar.gz](https://pan.baidu.com/s/1ZPIiOM__r1TRlmY4YGlolw) with code: `p72z`
5. uncompress data file using `tar -zxvf data.tar.gz`
6. modify the term `ctx` in `config/PEMS03/individual_GLU_mask_emb.json` to match your GPU devices
7. run code using `docker run -ti --rm --runtime=nvidia -v $PWD:/mxnet stsgcn/mxnet_1.41_cu100 python3 main.py --config config/PEMS03/individual_GLU_mask_emb.json`

If you are using Microsoft OpenPAI, modify the configurations saved in the folder `pai_jobs` to train STSGCNs on your clusters.

# repo structure
name|description
-|-
config|configurations of STSGCN
docker|dockerfile
models|core of STSGCN
pai_job|Microsoft OpenPAI configurations
paper|paper of STSGCN
test|pytest files
load_params.py|read parameters from local files
main.py|code of training STSGCN
pytest.ini|pytest configurations
requirements.txt|python packages requirements
utils.py|tools
