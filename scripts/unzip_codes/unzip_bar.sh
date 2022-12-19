echo "Unzip bar dataset"

zip_file=./dataset/bar.zip
save_dir=./dataset/bar/
mkdir $save_dir
unzip $zip_file -d ./dataset/
rm $zip_file