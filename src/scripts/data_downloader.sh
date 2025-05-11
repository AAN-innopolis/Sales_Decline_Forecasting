#run using bash (windows). Other operating systems dont know :)
# working directory /scripts

mkdir -p data/raw
cd data/raw || exit

curl -L -o liquor-sales.zip\
  https://www.kaggle.com/api/v1/datasets/download/almazkhannanov/liquor-sales

unzip liquor-sales.zip

rm liquor-sales.zip

