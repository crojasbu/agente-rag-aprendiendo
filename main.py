import os
from dotenv import load_dotenv

# 1. Cargar el archivo .env (leer el diario íntimo)
load_dotenv()

# 2. Obtener la clave (buscar en el diario)
mi_token = os.getenv("MI_CLAVE_SECRETA")
api_key = os.getenv("OPENAI_API_KEY")

# 3. Usarla (Imprimimos solo una parte para verificar)
print(f"El token cargado es: {mi_token}")

if api_key:
    print("¡La API Key de OpenAI se cargó correctamente!")
else:
    print("Error: No encontré la API Key.")