echo "Unzip bffhq dataset"

zip_file=./dataset/bffhq.zip
save_dir=./dataset/bffhq/
mkdir $save_dir
unzip $zip_file -d ./dataset/
rm $zip_file