sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install redis-server
pip install --upgrade pip
pip install uv
npm install -g localtunnel

uv sync

if [ ! -d "./data" ]; then
    chmod +x ./src/scripts/data_downloader.sh 
    ./src/scripts/data_downloader.sh
fi

./manage.sh start-all
./manage.sh status
lsof -i -P -n | grep LISTEN
