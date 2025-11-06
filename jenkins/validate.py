import os
import sys

def check_folder_structure():
    required_folders = [
        "src", "tests", "data", "jenkins", "models"
    ]
    missing = [f for f in required_folders if not os.path.exists(f)]
    if missing:
        print(f"❌ Carpetas faltantes: {missing}")
        sys.exit(1)
    print("✅ Estructura de carpetas verificada.")

def check_pyproject():
    print("Verificando pyproject.toml...")
    if not os.path.exists("pyproject.toml"):
        print("❌ Falta pyproject.toml")
        sys.exit(1)
    print("✅ pyproject.toml encontrado.")

if __name__ == "__main__":
    print("Iniciando validaciones previas al pipeline...")
    check_folder_structure()
    check_pyproject()
    print("✅ Validaciones completadas exitosamente.")
