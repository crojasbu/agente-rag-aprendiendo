import os
# --- LIMPIEZA DE RUIDO ---
# Desactivamos LangSmith para que no pida API Key
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Filtramos advertencias molestas
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# Imports Clásicos (Ahora funcionarán porque bajamos la versión)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

# 1. Cargar Claves
load_dotenv()

# Verificación de seguridad
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: Falta la OPENAI_API_KEY en el .env")
    exit()

# ==========================================
# FASE 1: RAG (La Memoria)
# ==========================================
print("--- 📚 1. Creando la base de conocimiento (RAG) ---")

texto_base = """
Data & AI Lead | Data Governance & Analytics
Profesional con +10 años de experiencia en banca líder del Perú (BCP, Pichincha), especializado en escalar capacidades
analíticas mediante Gobierno de Datos robusto e Inteligencia Artificial aplicada. Conecto Negocio, Riesgos y
Operaciones con soluciones tecnológicas que garantizan calidad, linaje y disponibilidad del dato crítico, combinando
visión estratégica con ejecución técnica."""

# Tokenizar (Cortar)
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documentos = text_splitter.create_documents([texto_base])

# Embedding (Vectorizar)
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(documentos, embeddings)

# ==========================================
# FASE 2: AGENTE (El Cerebro)
# ==========================================
print("--- 🧠 2. Configurando el Agente ---")

# Convertir la BD en una Herramienta
buscador = vector_db.as_retriever()
tool_buscador = create_retriever_tool(
    buscador,
    "info_cv",
    "Busca información sobre el CV de Carlos Rojas"
)
tools = [tool_buscador]

# Cerebro
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.8)

# Instrucciones
prompt = hub.pull("hwchase17/openai-functions-agent")

# Ensamblaje
agente = create_openai_functions_agent(llm, tools, prompt)
agente_ejecutor = AgentExecutor(agent=agente, tools=tools, verbose=True)

# ==========================================
# FASE 3: EJECUCIÓN
# ==========================================
print("\n--- 🚀 3. Ejecutando Agente ---\n")

pregunta = "Lista 5 caracteristicas que hacen match con el puesto de IA Lead? escribelo de manera sarcastica"
print(f"Usuario: {pregunta}\n")

respuesta = agente_ejecutor.invoke({"input": pregunta})

print("\n--- ✅ RESPUESTA FINAL ---")
print(respuesta["output"])