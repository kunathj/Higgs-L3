wget --no-check-certificate -O data.zip ONEDRIVE_DATA_LINK "https://onedrive.live.com/download?cid=D6A93F9F43A97047&resid=D6A93F9F43A97047%2167409&authkey=AD2t1BCEPMQgQSU"
unzip data.zip
rm data.zip

mkdir tmp
mkdir plots

conda env create -f environment.yml