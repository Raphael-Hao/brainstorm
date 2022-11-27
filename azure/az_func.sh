az_download() {
    mkdir -p "$2"
    azcopy sync "https://projectbrainstorm.blob.core.windows.net/largedata/$1/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" "$2" --recursive
}
az_upload() { azcopy sync "$1" "https://projectbrainstorm.blob.core.windows.net/largedata/$2/?sp=racwdli&st=2022-09-23T02:27:48Z&se=2023-02-01T10:27:48Z&spr=https&sv=2021-06-08&sr=c&sig=e%2B%2F9PGQi5%2B8g4bXsuJ14AHrk98RohKx51QFBlAWwIWg%3D" --recursive; }
