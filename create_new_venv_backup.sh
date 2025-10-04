# Save the current working directory
ORIGINAL_DIR=$(pwd)

# Automatically deactivate any active virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment ($VIRTUAL_ENV)..."
    deactivate
fi

# Check if required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <new_venv_name> <requirements_file_path>"
    return 1
fi

NEW_VENV_NAME="$1"
REQUIREMENTS_FILE="$2"
BASE_VENV_PATH="/Users/jethroestrada/.cache/venv/base_venv"
NEW_VENV_PATH="/Users/jethroestrada/.cache/venv/$NEW_VENV_NAME"

# Validate base_venv exists
if [ ! -d "$BASE_VENV_PATH" ]; then
    echo "Error: Base virtual environment not found at $BASE_VENV_PATH"
    return 1
fi

# Validate requirements file exists and is not empty
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file not found at $REQUIREMENTS_FILE"
    return 1
fi
if [ ! -s "$REQUIREMENTS_FILE" ]; then
    echo "Error: Requirements file $REQUIREMENTS_FILE is empty"
    return 1
fi

# Create new venv inheriting base_venv packages
python -m venv "$NEW_VENV_PATH"

# Validate Python version compatibility
BASE_PYTHON_VERSION=$(ls "$BASE_VENV_PATH/lib" | grep '^python[0-9]\.[0-9]*$')
NEW_PYTHON_VERSION=$(ls "$NEW_VENV_PATH/lib" | grep '^python[0-9]\.[0-9]*$')
if [ "$BASE_PYTHON_VERSION" != "$NEW_PYTHON_VERSION" ]; then
    echo "Error: Python version mismatch (base: $BASE_PYTHON_VERSION, new: $NEW_PYTHON_VERSION)"
    rm -rf "$NEW_VENV_PATH"
    return 1
fi

# Copy site-packages from base_venv to new_venv
rsync -a --exclude='pip*' --exclude='setuptools*' --exclude='__pycache__' \
    "$BASE_VENV_PATH/lib/$BASE_PYTHON_VERSION/site-packages/" \
    "$NEW_VENV_PATH/lib/$NEW_PYTHON_VERSION/site-packages/" || {
    echo "Error: Failed to copy site-packages from $BASE_VENV_PATH"
    rm -rf "$NEW_VENV_PATH"
    return 1
}

# Activate and install additional packages
source "$NEW_VENV_PATH/bin/activate"
pip install -r "$REQUIREMENTS_FILE"
# Verify installation
python -c "import sys; print('Virtual environment created successfully at:', sys.prefix)"
deactivate

# Restore original working directory
cd "$ORIGINAL_DIR" || echo "Warning: Failed to return to original directory $ORIGINAL_DIR"

# Usage Examples:
# 1. Create a new virtual environment named 'myproject' using a requirements file:
#    ./create_new_venv.sh myproject /path/to/requirements.txt
#
# 2. Create a virtual environment for a Django project with specific dependencies:
#    ./create_new_venv.sh django_env /home/user/projects/django_requirements.txt
#
# 3. Create a virtual environment for a data science project:
#    ./create_new_venv.sh data_science /home/user/requirements_data.txt
