echo "Configuring local environment for development..."

if exist venv\ (
    echo "Virtual environment already exists."
) else (
    python -m venv venv
)

echo "Activating virtual environment..."
call .\venv\Scripts\activate.bat

echo "Installing packages..."
call pip install --upgrade pip
call pip install -U poetry
call poetry install --no-root

echo "Ready for development!"
