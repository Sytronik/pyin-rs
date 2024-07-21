cargo build -F "build-binary $@" --release --bin pyin
gcc -Wall -g test/test.c -I./include -Ltarget/release/ -lpyin -o test_pyin
