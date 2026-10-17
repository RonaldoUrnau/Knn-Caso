import sys
from pathlib import Path

# Permite que las pruebas importen el paquete src/ cuando se ejecuta pytest desde la raíz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parent))
