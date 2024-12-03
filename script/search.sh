source='./data/data'
data='laion-10M'
C=4096
B=512
D=512
k=1

g++ -march=core-avx2 -Ofast -o ./bin/search_${data} ./src/search.cpp -I ./src/ -I /usr/include/eigen3 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

result_path=./results
mkdir ${result_path}

res="${result_path}/${data}/"

mkdir "$result_path/${data}/"

./bin/search_${data} -d ${data} -r ${res} -k ${k} -s "$source/$data/"
