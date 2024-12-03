C=4096
data='laion-10M'
D=512
B=512
source='./data/data'

g++ -fsanitize=address -o ./bin/index_${data} ./src/index.cpp -I ./src/ -I /usr/include/eigen3 -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN

./bin/index_${data} -d $data -s "$source/$data/"    