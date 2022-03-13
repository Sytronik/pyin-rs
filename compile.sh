cargo build --release
gcc -Wall -g test/test.c -I./include -Ltarget/release/ -lpyin -o test_pyin
