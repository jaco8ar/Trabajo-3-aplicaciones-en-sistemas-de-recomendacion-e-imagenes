echo "Actualizando pip..."
pip3 install --upgrade pip

echo "Instalando dependencias desde requirements.txt..."
pip3 install -r requirements.txt

echo "Listo. Ahora puedes ejecutar:"
echo "streamlit run app.py"