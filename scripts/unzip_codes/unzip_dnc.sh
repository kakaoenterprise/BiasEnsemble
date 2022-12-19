echo "Unzip dogs&cats dataset"

zip_file=./dataset/dogs_and_cats.zip
save_dir=./dataset/dogs_and_cats/
mkdir $save_dir
unzip $zip_file -d ./dataset/
rm $zip_file