echo "Unzip cmnist dataset"

zip_file=./dataset/cmnist.zip
save_dir=./dataset/cmnist/
mkdir $save_dir
unzip $zip_file -d ./dataset/
rm $zip_file