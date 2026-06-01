#  🧠 SISTEMA REDENTOR v7.3-L
#  Copyright (C) 2026 Jose Miguel Vargas Alvarez. Todos los derechos reservados.
#  
#  AVISO LEGAL: Este software está protegido por una licencia propietaria no comercial. 
#  Se otorga derecho de uso temporal y revocable exclusivamente para fines personales o educativos.
#  QUE DAR TERMINANTEMENTE PROHIBIDO SU USO COMERCIAL O DISTRIBUCIÓN SIN AUTORIZACIÓN.
#  Consulte el archivo 'LICENSE' adjunto para más detalles.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zipfile
import io
import os
import json
import time
import sys
import requests
import threading
from collections import OrderedDict

# ---------------------------------------------------------------------------
# VARIABLES Y FUNCIONES DE CONTROL GLOBAL (Teclado y Control de Parada)
# ---------------------------------------------------------------------------
stop_training = False

def escuchador_teclado():
    """Hilo secundario para capturar la detención manual sin pausar el entrenamiento."""
    global stop_training
    while not stop_training:
        try:
            linea = input().strip().lower()
            if linea == 'd':
                print("\n\n[Teclado] Solicitando parada segura. Guardando al terminar el ciclo actual...")
                stop_training = True
                break
        except:
            break

def obtener_tamano_total_db(urls, headers):
    """Calcula el tamaño total en caracteres/bytes de toda la base de datos remota."""
    total = 0
    for url in urls:
        try:
            r = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
            if "Content-Length" in r.headers:
                total += int(r.headers["Content-Length"])
            else:
                raise Exception()
        except:
            # Tamaños reales de respaldo si hay bloqueos de conexión o falta de cabeceras
            if "es_corpus" in url: total += 11489000000  # Escorpius aa (~10.7 GB)
            elif "pg2000" in url: total += 5400000        # Don Quijote / Gutenberg (~5.1 MB)
            else: total += 15700000                        # RAE Corpus (~15 MB)
    return total

# ---------------------------------------------------------------------------
# 1. TOKENIZADOR ULTRA-LIGERO A NIVEL DE CARACTERES
# ---------------------------------------------------------------------------
class TokenizadorCaracteres:
    def __init__(self, texto_o_lista):
        if isinstance(texto_o_lista, list):
            self.vocab = texto_o_lista
        else:
            caracteres = sorted(list(set(texto_o_lista)))
            if " " not in caracteres: caracteres.append(" ")
            if "\n" not in caracteres: caracteres.append("\n")
            self.vocab = caracteres
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def codificar(self, texto):
        idx_espacio = self.char2idx.get(" ", 0)
        return [self.char2idx.get(ch, idx_espacio) for ch in texto]

    def decodificar(self, indices):
        return "".join([self.idx2char.get(idx, " ") for idx in indices])

# ---------------------------------------------------------------------------
# 2. STREAMER DE DATOS RESILIENTE
# ---------------------------------------------------------------------------
class RedentorStreamer:
    def __init__(self, urls):
        self.urls = urls
        self.current_url_idx = 0
        self.buffer = ""
        self.response = None
        self.iterator = None
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    def _get_next_stream(self):
        if self.current_url_idx >= len(self.urls):
            self.current_url_idx = 0
        url = self.urls[self.current_url_idx]
        try:
            print(f"\n[Streaming] Conectando a fuente [{self.current_url_idx + 1}/{len(self.urls)}]: {url.split('/')[-1]}...")
            self.response = requests.get(url, stream=True, timeout=15, headers=self.headers)
            self.response.raise_for_status()
            self.iterator = self.response.iter_lines(decode_unicode=True)
            self.current_url_idx += 1
        except Exception as e:
            print(f"[Error] Conexión fallida a {url.split('/')[-1]}: {e}. Reintentando en 5s...")
            time.sleep(5)
            self.iterator = None

    def get_data(self, min_chars=1000):
        while len(self.buffer) < min_chars:
            try:
                if self.iterator is None: self._get_next_stream()
                if self.iterator is None: continue
                line = next(self.iterator)
                if isinstance(line, bytes): line = line.decode('utf-8', errors='ignore')
                if line and len(line.strip()) > 5: self.buffer += line.strip() + " \n "
            except StopIteration:
                self.iterator = None
                continue
        chunk = self.buffer[:min_chars]
        self.buffer = self.buffer[min_chars:]
        return chunk

# ---------------------------------------------------------------------------
# 3. KERNELS Y ARQUITECTURA MoE (Piloto-Copiloto)
# ---------------------------------------------------------------------------
@torch.jit.script
def fusion_kernel(h_p, h_c, w_p, w_c, bias):
    alpha = torch.sigmoid(w_p * h_p + w_c * h_c + bias)
    return alpha * h_p + (1 - alpha) * h_c

@torch.jit.script
def compute_output(h_flujo, mascara, h_p, h_fused):
    peso_c = torch.sigmoid(mascara).unsqueeze(0)
    return (1 - peso_c) * h_p + peso_c * h_fused

class ParamStoreV7(nn.Module):
    def __init__(self, num_neuronas, num_departamentos, max_colas=4, ruta_base="modelo_redentor/departamentos"):
        super().__init__()
        self.num_neuronas, self.num_departamentos, self.max_colas, self.ruta_base = num_neuronas, num_departamentos, max_colas, ruta_base
        self.cache = OrderedDict()
        self.max_ram = 16
        self.pesos_deps = nn.ParameterList([
            nn.Parameter(torch.empty(num_neuronas, num_neuronas + 1)) for _ in range(max_colas * num_departamentos)
        ])
        for p in self.pesos_deps:
            nn.init.xavier_uniform_(p[:, :num_neuronas])
            p.data[:, -1] = 0.0

    def _get_indice_maestro(self, id_dep, cola): return (cola - 1) * self.num_departamentos + id_dep
    def _get_ruta_archivo(self, id_dep, cola): return os.path.join(self.ruta_base, f"cola{cola}_dpto{id_dep}.dpn")

    def cargar_a_cache(self, id_dep, cola, device='cpu'):
        clave_cache = f"cola_{cola}_dep_{id_dep}"
        if clave_cache in self.cache:
            self.cache.move_to_end(clave_cache)
            return self.cache[clave_cache]
        if len(self.cache) >= self.max_ram: self.cache.popitem(last=False)
        ruta = self._get_ruta_archivo(id_dep, cola)
        if os.path.exists(ruta):
            with zipfile.ZipFile(ruta, 'r') as zf:
                buffer = io.BytesIO(zf.read("data.pt"))
                estado = torch.load(buffer, map_location=device, weights_only=False)
            params = (estado["capa_oculta"]["data"].float() - estado["capa_oculta"]["zero_point"]) * estado["capa_oculta"]["scale"]
        else:
            params = self.pesos_deps[self._get_indice_maestro(id_dep, cola)].detach().to(device)
        self.cache[clave_cache] = params
        return params

class CapaOcultaV7(nn.Module):
    def __init__(self, num_neuronas, num_departamentos, param_store):
        super().__init__()
        self.num_neuronas, self.num_departamentos, self.param_store = num_neuronas, num_departamentos, param_store
        self.w_p_f, self.w_c_f, self.bias_f = nn.Parameter(torch.randn(num_neuronas)), nn.Parameter(torch.randn(num_neuronas)), nn.Parameter(torch.zeros(num_neuronas))
        self.mascara_copiloto = nn.Parameter(torch.rand(num_neuronas))

    def forward(self, h_flujo, ids_p=None, ids_c=None, probs_p=None, probs_c=None, cola=1):
        device = h_flujo.device
        if self.training:
            h_deps = []
            for d in range(self.num_departamentos):
                idx = self.param_store._get_indice_maestro(d, cola)
                p_params = self.param_store.pesos_deps[idx].to(device)
                h_deps.append(F.relu(F.linear(h_flujo, p_params[:, :self.num_neuronas], p_params[:, -1])))
            h_all = torch.stack(h_deps, dim=1)
            h_p = torch.sum(h_all * probs_p.unsqueeze(-1), dim=1)
            h_c = torch.sum(h_all * probs_c.unsqueeze(-1), dim=1)
            h_fused = fusion_kernel(h_p, h_c, self.w_p_f, self.w_c_f, self.bias_f)
            return compute_output(h_flujo, self.mascara_copiloto, h_p, h_fused)
        else:
            salida = torch.zeros(h_flujo.size(0), self.num_neuronas, device=device)
            for p_id in ids_p.unique():
                mask_p = (ids_p == p_id)
                p_params = self.param_store.cargar_a_cache(p_id.item(), cola, device=device)
                h_p = F.relu(F.linear(h_flujo[mask_p], p_params[:, :self.num_neuronas], p_params[:, -1]))
                ids_c_sub = ids_c[mask_p]
                for c_id in ids_c_sub.unique():
                    mask_c = (ids_c_sub == c_id)
                    c_params = self.param_store.cargar_a_cache(c_id.item(), cola, device=device)
                    h_c = F.relu(F.linear(h_flujo[mask_p][mask_c], c_params[:, :self.num_neuronas], c_params[:, -1]))
                    h_fused = fusion_kernel(h_p[mask_c], h_c, self.w_p_f, self.w_c_f, self.bias_f)
                    m_global = mask_p.clone(); m_global[mask_p] = mask_c
                    salida[m_global] = compute_output(h_flujo[m_global], self.mascara_copiloto, h_p[mask_c], h_fused)
            return salida

class ModeloHablaDepartamentalV7(nn.Module):
    def __init__(self, vocab_size, context_length=16, dim_emb=64, num_neuronas=128, num_departamentos=8, max_colas=4, nombre_modelo="modelo_redentor"):
        super().__init__()
        self.vocab_size, self.context_length, self.num_neuronas, self.num_departamentos, self.max_colas, self.nombre_modelo = vocab_size, context_length, num_neuronas, num_departamentos, max_colas, nombre_modelo
        self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.proyeccion_entrada = nn.Linear(context_length * dim_emb, num_neuronas)
        self.enrutador = nn.Sequential(nn.Linear(context_length * dim_emb, 64), nn.ReLU(), nn.Linear(64, num_departamentos * 2))
        self.param_store = ParamStoreV7(num_neuronas, num_departamentos, max_colas, ruta_base=os.path.join(nombre_modelo, "departamentos"))
        self.capa_oculta = CapaOcultaV7(num_neuronas, num_departamentos, self.param_store)
        self.fc_salida = nn.Linear(num_neuronas, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        emb = self.embedding(x).view(batch_size, -1)
        logits_enr = self.enrutador(emb)
        logits_p, logits_c = logits_enr[:, :self.num_departamentos], logits_enr[:, self.num_departamentos:]
        if self.training:
            probs_p = F.gumbel_softmax(logits_p, tau=1.0, hard=True)
            probs_c = F.gumbel_softmax(logits_c, tau=1.0, hard=True)
            idx_p, idx_c = torch.argmax(probs_p, dim=1), torch.argmax(probs_c, dim=1)
        else:
            idx_p, idx_c = torch.argmax(logits_p, dim=1), torch.argmax(logits_c, dim=1)
            probs_p, probs_c = None, None
        h_flujo = F.relu(self.proyeccion_entrada(emb))
        for c in range(1, self.max_colas + 1):
            h_transformado = self.capa_oculta(h_flujo, ids_p=idx_p, ids_c=idx_c, probs_p=probs_p, probs_c=probs_c, cola=c)
            h_flujo = h_flujo + h_transformado
        return self.fc_salida(h_flujo)

    def save_modelo(self):
        ruta = self.nombre_modelo
        if not os.path.exists(ruta): os.makedirs(ruta)
        for c in range(1, self.max_colas + 1):
            for d in range(self.num_departamentos):
                self._guardar_dpn(d, c, self.param_store.pesos_deps[self.param_store._get_indice_maestro(d, c)])
        torch.save({
            "embedding": self.embedding.state_dict(), "proyeccion_entrada": self.proyeccion_entrada.state_dict(),
            "enrutador": self.enrutador.state_dict(), "capa_oculta": {"w_p_f": self.capa_oculta.w_p_f, "w_c_f": self.capa_oculta.w_c_f, "bias_f": self.capa_oculta.bias_f, "mascara": self.capa_oculta.mascara_copiloto},
            "fc_salida": self.fc_salida.state_dict(), "config": {"vs": self.vocab_size, "cl": self.context_length, "nh": self.num_neuronas, "nd": self.num_departamentos, "mc": self.max_colas, "nm": self.nombre_modelo, "vocab": self.tokenizador.vocab if hasattr(self, "tokenizador") else None}
        }, os.path.join(ruta, "base_redentor.pt"))

    def _guardar_dpn(self, id_dep, cola, params):
        if params.numel() == 0: return
        ruta = self.param_store._get_ruta_archivo(id_dep, cola)
        if not os.path.exists(os.path.dirname(ruta)): os.makedirs(os.path.dirname(ruta))
        p_min, p_max = params.min().item(), params.max().item()
        scale = (p_max - p_min) / 255.0 if (p_max - p_min) > 0 else 1.0
        zero_point = max(0.0, min(255.0, float(round(-p_min / scale))))
        q_data = torch.round((params / scale) + zero_point).clamp(0, 255).to(torch.uint8)
        buffer = io.BytesIO(); torch.save({"capa_oculta": {"data": q_data, "scale": scale, "zero_point": zero_point}}, buffer)
        buffer.seek(0)
        with zipfile.ZipFile(ruta, 'w', zipfile.ZIP_DEFLATED) as zf: zf.writestr("data.pt", buffer.read())

    @classmethod
    def load_modelo(cls, ruta="modelo_redentor", device='cpu', para_entrenar=False):
        base = torch.load(os.path.join(ruta, "base_redentor.pt"), map_location=device, weights_only=False)
        cfg = base["config"]
        m = cls(cfg["vs"], cfg["cl"], dim_emb=64, num_neuronas=cfg["nh"], num_departamentos=cfg["nd"], max_colas=cfg["mc"], nombre_modelo=cfg["nm"]).to(device)
        m.embedding.load_state_dict(base["embedding"]); m.proyeccion_entrada.load_state_dict(base["proyeccion_entrada"])
        m.enrutador.load_state_dict(base["enrutador"]); m.capa_oculta.w_p_f.data = base["capa_oculta"]["w_p_f"].to(device)
        m.capa_oculta.w_c_f.data = base["capa_oculta"]["w_c_f"].to(device); m.capa_oculta.bias_f.data = base["capa_oculta"]["bias_f"].to(device)
        m.capa_oculta.mascara_copiloto.data = base["capa_oculta"]["mascara"].to(device); m.fc_salida.load_state_dict(base["fc_salida"])
        if "vocab" in cfg and cfg["vocab"]: m.tokenizador = TokenizadorCaracteres(cfg["vocab"])
        
        if para_entrenar:
            print("[Carga] Reconstruyendo expertos en FP32 para entrenamiento continuo...")
            for c in range(1, m.max_colas + 1):
                for d in range(m.num_departamentos):
                    idx = m.param_store._get_indice_maestro(d, c)
                    ruta_dpn = m.param_store._get_ruta_archivo(d, c)
                    if os.path.exists(ruta_dpn):
                        with zipfile.ZipFile(ruta_dpn, 'r') as zf:
                            estado = torch.load(io.BytesIO(zf.read("data.pt")), map_location=device, weights_only=False)
                        params = (estado["capa_oculta"]["data"].float() - estado["capa_oculta"]["zero_point"]) * estado["capa_oculta"]["scale"]
                        m.param_store.pesos_deps[idx] = nn.Parameter(params.to(device), requires_grad=True)
                    else:
                        m.param_store.pesos_deps[idx].requires_grad = True
        else:
            for i in range(len(m.param_store.pesos_deps)): 
                m.param_store.pesos_deps[i] = nn.Parameter(torch.empty(0), requires_grad=False)
        return m

def generar_texto(modelo, tokenizador, semilla, longitud=100, temperatura=0.7):
    modelo.eval(); context_len = modelo.context_length; resultado = semilla
    device = next(modelo.embedding.parameters()).device
    with torch.no_grad():
        for _ in range(longitud):
            texto_contexto = resultado[-context_len:].rjust(context_len, " ")
            indices = tokenizador.codificar(texto_contexto)
            logits = modelo(torch.tensor([indices], dtype=torch.long).to(device))
            probs = F.softmax(logits[0] / temperatura, dim=-1)
            resultado += tokenizador.decodificar([torch.multinomial(probs, num_samples=1).item()])
    return resultado

# ---------------------------------------------------------------------------
# 4. MENÚ INTERACTIVO Y BUCLE DE ENTRENAMIENTO POR TEMPORIZADOR TÉRMICO
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    NOMBRE_MODELO = "modelo_redentor_v1"
    RUTA_CHECKPOINT = os.path.join(NOMBRE_MODELO, "checkpoint_stream.json")
    URLS = [
        "https://huggingface.co/datasets/LHF/escorpius/resolve/main/es_corpus.jsonl.aa",
        "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
        "https://raw.githubusercontent.com/eneko98/RAE-Corpus/master/RealAcademiaEspanola-DiccionarioLlenguaEspanola.txt"
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char_base = sorted(list(set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áéíóúñÁÉÍÓÚÑ¿¡.,:;()!?- \"\n\t_")))
    tokenizador_nuevo = TokenizadorCaracteres(char_base)

    modelo = None
    if os.path.exists(os.path.join(NOMBRE_MODELO, "base_redentor.pt")):
        try:
            modelo = ModeloHablaDepartamentalV7.load_modelo(NOMBRE_MODELO, device=device, para_entrenar=False)
            tokenizador = modelo.tokenizador
        except: modelo = None

    if modelo is None:
        tokenizador = tokenizador_nuevo
        modelo = ModeloHablaDepartamentalV7(tokenizador.vocab_size, 16, 64, 128, 8, 4, NOMBRE_MODELO).to(device)
        modelo.tokenizador = tokenizador

    while True:
        print("\n" + "═"*45)
        print("      🧠 SISTEMA REDENTOR v7.3-L")
        print("═"*45)
        print("  1. ENTRENAR (Corrido - Ciclo Térmico 10min/5min)")
        print("  2. HABLAR (Probar lo aprendido)")
        print("  3. SALIR")
        print("─"*45)
        opcion = input(" Selecciona una opción: ").strip()

        if opcion == "1":
            stop_training = False
            
            print("\n[INFO] Evaluando base de datos global distribuida...")
            tamano_total_db = obtener_tamano_total_db(URLS, {'User-Agent': 'Mozilla/5.0'})
            
            print("[INFO] Restaurando matrices de expertos...")
            if os.path.exists(os.path.join(NOMBRE_MODELO, "base_redentor.pt")):
                try:
                    modelo = ModeloHablaDepartamentalV7.load_modelo(NOMBRE_MODELO, device=device, para_entrenar=True)
                    tokenizador = modelo.tokenizador
                except Exception as e:
                    print(f"[Aviso] Error restaurando: {e}. Usando esqueleto base.")

            posicion = {"url_idx": 0, "chars": 0}
            if os.path.exists(RUTA_CHECKPOINT):
                with open(RUTA_CHECKPOINT, "r") as f: posicion = json.load(f)
                print(f"[INFO] Reanudando histórico. Procesados: {posicion['chars']} caracteres.")

            optimizer = torch.optim.AdamW(modelo.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            streamer = RedentorStreamer(URLS)
            streamer.current_url_idx = posicion["url_idx"]
            if posicion["chars"] > 0: streamer.get_data(posicion["chars"] % 5000); streamer.buffer = ""

            # Activar escucha de teclado en segundo plano ("d" + Enter)
            hilo_teclado = threading.Thread(target=escuchador_teclado, daemon=True)
            hilo_teclado.start()

            modelo.train()
            print("\n" + "🔥"*25)
            print(" MODO CRONÓMETRO TÉRMICO ACTIVO")
            print(" - Entrenamiento continuo: 10 Minutos")
            print(" - Descanso de enfriamiento: 5 Minutos")
            print(" - Detener proceso seguro: Escribe 'd' + Enter")
            print("" + "🔥"*25 + "\n")

            try:
                total_loss, steps = 0, 0
                context_len = modelo.context_length
                
                # Configuración de tiempos exactos en segundos
                TIEMPO_ENTRENAMIENTO = 10 * 60  
                TIEMPO_ENFRIAMIENTO = 5 * 60    
                
                proxima_pausa = time.time() + TIEMPO_ENTRENAMIENTO
                
                while not stop_training:
                    # 1. Comprobación y activación de Pausa por Enfriamiento
                    if time.time() >= proxima_pausa:
                        print(f"\n\n[PAUSA DE ENFRIAMIENTO] Cuidando procesador... Descanso preventivo de 5 minutos activo.")
                        
                        # Guardado preventivo inmediato
                        modelo.save_modelo()
                        posicion["url_idx"] = streamer.current_url_idx
                        with open(RUTA_CHECKPOINT, "w") as f: json.dump(posicion, f)
                        
                        # Espera total de enfriamiento controlando si el usuario presiona 'd'
                        inicio_pausa = time.time()
                        while time.time() - inicio_pausa < TIEMPO_ENFRIAMIENTO:
                            if stop_training: break
                            time.sleep(1)
                        
                        if stop_training: break
                        
                        print("\n[REANUDANDO] Procesador refrigerado con éxito. Continuando entrenamiento...\n")
                        proxima_pausa = time.time() + TIEMPO_ENTRENAMIENTO
                    
                    # 2. Obtener fragmento de la base de datos
                    texto = streamer.get_data(2000)
                    posicion["chars"] += len(texto)
                    indices = tokenizador.codificar(texto)
                    X, Y = [], []
                    for i in range(len(indices) - context_len):
                        X.append(indices[i:i+context_len]); Y.append(indices[i+context_len])
                    if not X: continue
                    
                    dl = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)), batch_size=64, shuffle=True)
                    
                    # CORRECCIÓN: 8 repasos locales por bloque
                    for epoch_local in range(8):
                        if stop_training: break
                        for bx, by in dl:
                            if stop_training: break
                            bx, by = bx.to(device), by.to(device)
                            optimizer.zero_grad()
                            logits = modelo(bx)
                            loss = criterion(logits, by)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                            steps += 1
                    
                    # 3. Calcular porcentaje real completado de toda la Base de Datos
                    porcentaje_avance = (posicion["chars"] / tamano_total_db) * 100
                    
                    # Calcular minutos para el descanso actual
                    minutos_restantes = max(0, int((proxima_pausa - time.time()) // 60))
                    segundos_restantes = max(0, int((proxima_pausa - time.time()) % 60))
                    
                    # Muestra limpia en consola (Porcentaje exacto, pérdida y reloj térmico inverso)
                    sys.stdout.write(f"\r[Progreso DB] {porcentaje_avance:.4f}% | Loss: {total_loss/(steps if steps>0 else 1):.4f} | Enfriamiento en: {minutos_restantes:02d}:{segundos_restantes:02d} ")
                    sys.stdout.flush()

                # Cierre y guardado estructurado al presionar "d"
                print("\n[Guardando] Volcando pesos a archivos comprimidos .dpn e INT8...")
                modelo.save_modelo()
                posicion["url_idx"] = streamer.current_url_idx
                with open(RUTA_CHECKPOINT, "w") as f: json.dump(posicion, f)
                print("[Éxito] Progreso e historial consolidados perfectamente.")
                
            except KeyboardInterrupt:
                print("\n[Interrupción] Forzando guardado de emergencia...")
                modelo.save_modelo()
                posicion["url_idx"] = streamer.current_url_idx
                with open(RUTA_CHECKPOINT, "w") as f: json.dump(posicion, f)

        elif opcion == "2":
            print("\n[INFO] Cargando modelo en modo inferencia ligera...")
            if os.path.exists(os.path.join(NOMBRE_MODELO, "base_redentor.pt")):
                try:
                    modelo = ModeloHablaDepartamentalV7.load_modelo(NOMBRE_MODELO, device=device, para_entrenar=False)
                    tokenizador = modelo.tokenizador
                except Exception as e:
                    print(f"[Error] No se pudo cargar para chat: {e}")
                    continue

            print("\n💬 MODO CHAT ACTIVO (Escribe 'volver' para salir)")
            modelo.eval()
            while True:
                usuario = input("\nTú: ").strip()
                if usuario.lower() in ["volver", "salir"]: break
                if not usuario: continue
                print("Redentor: ", end="", flush=True)
                respuesta = generar_texto(modelo, tokenizador, usuario, longitud=80, temperatura=0.6)
                print(respuesta[len(usuario):])

        elif opcion == "3":
            print("\n[Cierre] Sistema Redentor fuera de línea.")
            break
