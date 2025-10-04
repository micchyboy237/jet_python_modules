PY_VERSION=3.12.9

pyenv uninstall -f $PY_VERSION
env \
  LDFLAGS="-L$(brew --prefix zlib)/lib -L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib" \
  CPPFLAGS="-I$(brew --prefix zlib)/include -I$(brew --prefix openssl)/include -I$(brew --prefix readline)/include" \
  pyenv install -v $PY_VERSION
pip install -U pip

echo "Verifying Python and pip versions..."
python --version
pip --version

echo "✅ Python $PY_VERSION successfully reinstalled and pip upgraded!"
