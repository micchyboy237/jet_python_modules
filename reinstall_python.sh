pyenv uninstall -f 3.12.9
env LDFLAGS="-L$(brew --prefix zlib)/lib -L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib" \
    CPPFLAGS="-I$(brew --prefix zlib)/include -I$(brew --prefix openssl)/include -I$(brew --prefix readline)/include" \
    pyenv install 3.12.9
