sudo apt-get -y update
sudo apt-get -y upgrade
pip install --upgrade pip
pip install uv
npm install -g localtunnel

uv sync

./manage.sh start-all
./manage.sh status
lsof -i -P -n | grep LISTEN
