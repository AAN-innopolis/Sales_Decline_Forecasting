sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install redis-server
pip install --upgrade pip
pip install uv
npm install -g localtunnel

cd Sales_Decline_Forecasting
uv sync

if [ ! -d "./data" ]; then
  ./src/scripts/data_downloader.sh
fi

./manage.sh start-all
./manage.sh status
lsof -i -P -n | grep LISTEN
