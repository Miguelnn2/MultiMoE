#  🧠 SISTEMA REDENTOR v7.3-L
#  Copyright (C) 2026 Jose Miguel Vargas Alvarez. Todos los derechos reservados.
#  
#  AVISO LEGAL: Este software está protegido por una licencia propietaria no comercial. 
#  Se otorga derecho de uso temporal y revocable exclusivamente para fines personales o educativos.
#  QUE DAR TERMINANTEMENTE PROHIBIDO SU USO COMERCIAL O DISTRIBUCIÓN SIN AUTORIZACIÓN.
#  Consulte el archivo 'LICENSE' adjunto para más detalles.

"""
===============================================================================
MODELO NEURONAL DEPARTAMENTAL (V5 - MOE_G Optimizado para Producción)
===============================================================================
Comandos de Control en Inferencia:
/turbo : Modo Batch (Veloz). Un dpto por batch.
/fino  : Modo Muestra (Inteligente). Dpto por cada entrada.
/auto  : Gestión LRU dinámica de RAM.
/fijo  : Bloquea dptos en RAM para evitar lectura de disco.
/gordo : Activa el Mecanismo de Fusión Completo de la V2.

Mejora Clave V5:
- Poda de Artefactos de Entrenamiento (Guarda un esqueleto de producción y
  purga los pesos FP32 redundantes, reduciendo drásticamente el espacio en disco).
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import zipfile
import io
import os
import time
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 1. KERNELS OPTIMIZADOS (TorchScript para mitigar el overhead de Python)
# ---------------------------------------------------------------------------
@torch.jit.script
def fusion_kernel(h_p, h_c, w_p, w_c, bias):
    """Operación de fusión optimizada compilada en C++ nativo."""
    alpha = torch.sigmoid(w_p * h_p + w_c * h_c + bias)
    return alpha * h_p + (1 - alpha) * h_c

@torch.jit.script
def compute_output(x, mascara, h_p, h_c_fused):
    """Cálculo final fusionado de piloto y copiloto."""
    peso_c = torch.sigmoid(mascara).unsqueeze(0)
    return (1 - peso_c) * h_p + peso_c * h_c_fused

# ---------------------------------------------------------------------------
# 2. MECANISMO DE FUSIÓN PESADO (/gordo)
# ---------------------------------------------------------------------------
class MecanismoFusionPesado(nn.Module):
    def __init__(self, num_neuronas):
        super().__init__()
        self.w_piloto = nn.Parameter(torch.randn(num_neuronas))
        self.w_copiloto = nn.Parameter(torch.randn(num_neuronas))
        self.bias = nn.Parameter(torch.zeros(num_neuronas))

    def forward(self, h_p, h_c):
        alpha = torch.sigmoid(self.w_piloto * h_p + self.w_copiloto * h_c + self.bias)
        return alpha * h_p + (1 - alpha) * h_c

# ---------------------------------------------------------------------------
# 3. ALMACÉN DE PARÁMETROS HÍBRIDO (Gradientes en FP32 / Inferencia en INT8)
# ---------------------------------------------------------------------------
class ParamStore(nn.Module):
    def __init__(self, num_neuronas, dim_entrada, num_departamentos, ruta_base="modelo_departamental/departamentos"):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.dim_entrada = dim_entrada
        self.num_departamentos = num_departamentos
        self.ruta_base = ruta_base
        
        # Estructuras dinámicas de Inferencia
        self.cache = OrderedDict()
        self.fijos = set()
        self.max_ram = 3
        
        # Parámetros Maestros Contiguos (Obligatorios para Backpropagation en train)
        self.pesos_deps = nn.ParameterList([
            nn.Parameter(torch.empty(num_neuronas, dim_entrada + 1))
            for _ in range(num_departamentos)
        ])
        for p in self.pesos_deps:
            nn.init.xavier_uniform_(p[:, :dim_entrada])
            p.data[:, -1] = 0.0  # Inicializar sesgos en cero

    def _get_ruta(self, id_dep):
        nombres = ["mates", "lengua", "fisica", "logica", "historia"]
        return os.path.join(self.ruta_base, f"{nombres[id_dep]}.dpn")

    def cargar_a_cache(self, id_dep, device='cpu'):
        # MODO ENTRENAMIENTO: Devuelve el parámetro maestro directo para acumular gradientes
        if self.training:
            return self.pesos_deps[id_dep].to(device)
            
        # MODO INFERENCIA: Lógica ultra-ligera basada en caché LRU y descompresión INT8
        if id_dep in self.cache:
            if id_dep not in self.fijos:
                self.cache.move_to_end(id_dep)
            return self.cache[id_dep]
        
        if len(self.cache) >= self.max_ram and id_dep not in self.fijos:
            for k in list(self.cache.keys()):
                if k not in self.fijos:
                    self.cache.pop(k)
                    break
            
        ruta = self._get_ruta(id_dep)
        if os.path.exists(ruta):
            with zipfile.ZipFile(ruta, 'r') as zf:
                data = zf.read("data.pt")
                buffer = io.BytesIO(data)
                estado = torch.load(buffer, map_location=device, weights_only=False)
            
            q_data = estado["capa_oculta"]["data"]
            scale = estado["capa_oculta"]["scale"]
            zero_point = estado["capa_oculta"]["zero_point"]
            params = (q_data.float() - zero_point) * scale
        else:
            # Fallback por si no se han exportado los .dpn todavía
            params = self.pesos_deps[id_dep].detach().to(device)
            
        self.cache[id_dep] = params
        return params

    def fijar(self, id_dep):
        self.cargar_a_cache(id_dep)
        self.fijos.add(id_dep)

    def liberar_todo(self):
        self.fijos.clear()
        self.cache.clear()

# ---------------------------------------------------------------------------
# 4. CAPA DEPARTAMENTAL COOPERATIVA
# ---------------------------------------------------------------------------
class CapaDepartamental(nn.Module):
    def __init__(self, num_neuronas, dim_entrada, num_departamentos, param_store):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.dim_entrada = dim_entrada
        self.param_store = param_store
        
        self.w_p_f = nn.Parameter(torch.randn(num_neuronas))
        self.w_c_f = nn.Parameter(torch.randn(num_neuronas))
        self.bias_f = nn.Parameter(torch.zeros(num_neuronas))
        self.mascara_copiloto = nn.Parameter(torch.rand(num_neuronas))
        
        self.fusion_pesada = MecanismoFusionPesado(num_neuronas)

    def forward(self, x, ids_p, ids_c, modo_granularidad="fino", usar_copiloto=True, modo_gordo=False):
        device = x.device
        
        if modo_granularidad == "turbo" and not modo_gordo:
            id_p = torch.mode(ids_p).values.item()
            p_params = self.param_store.cargar_a_cache(int(id_p), device=device)
            h_p = F.relu(F.linear(x, p_params[:, :self.dim_entrada], p_params[:, -1]))
            
            if not usar_copiloto: return h_p
            
            id_c = torch.mode(ids_c).values.item()
            c_params = self.param_store.cargar_a_cache(int(id_c), device=device)
            h_c = F.relu(F.linear(x, c_params[:, :self.dim_entrada], c_params[:, -1]))
            
            h_fused = fusion_kernel(h_p, h_c, self.w_p_f, self.w_c_f, self.bias_f)
            return compute_output(x, self.mascara_copiloto, h_p, h_fused)

        else:
            salida = torch.zeros(x.size(0), self.num_neuronas, device=device)
            unique_p = ids_p.unique()
            for p_id in unique_p:
                mask_p = (ids_p == p_id)
                p_params = self.param_store.cargar_a_cache(p_id.item(), device=device)
                h_p = F.relu(F.linear(x[mask_p], p_params[:, :self.dim_entrada], p_params[:, -1]))
                
                if usar_copiloto:
                    ids_c_sub = ids_c[mask_p]
                    for c_id in ids_c_sub.unique():
                        mask_c = (ids_c_sub == c_id)
                        c_params = self.param_store.cargar_a_cache(c_id.item(), device=device)
                        h_c = F.relu(F.linear(x[mask_p][mask_c], c_params[:, :self.dim_entrada], c_params[:, -1]))
                        
                        if modo_gordo:
                            h_fused = self.fusion_pesada(h_p[mask_c], h_c)
                        else:
                            h_fused = fusion_kernel(h_p[mask_c], h_c, self.w_p_f, self.w_c_f, self.bias_f)
                        
                        h_final = compute_output(x[mask_p][mask_c], self.mascara_copiloto, h_p[mask_c], h_fused)
                        
                        m_global = mask_p.clone()
                        m_global[mask_p] = mask_c
                        salida[m_global] = h_final
                else:
                    salida[mask_p] = h_p
            return salida

# ---------------------------------------------------------------------------
# 5. MODELO COMPLETO CON CONEXIÓN RESIDUAL Y ESTIMADOR STE
# ---------------------------------------------------------------------------
class ModeloDepartamental(nn.Module):
    def __init__(self, dim_entrada, num_neuronas, num_departamentos, dim_salida):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.dim_entrada = dim_entrada
        self.num_departamentos = num_departamentos
        
        self.param_store = ParamStore(num_neuronas, dim_entrada, num_departamentos)
        self.enrutador = nn.Sequential(
            nn.Linear(dim_entrada, 32),
            nn.ReLU(),
            nn.Linear(32, num_departamentos * 2)
        )
        self.capa_oculta = CapaDepartamental(num_neuronas, dim_entrada, num_departamentos, self.param_store)
        self.fc_salida = nn.Linear(num_neuronas, dim_salida)
        
        # Conexión Residual de Estabilidad Global (Evita colapsos de gradiente)
        self.residual = nn.Linear(dim_entrada, dim_salida)
        
        self.modo_granularidad = "fino"
        self.modo_carga = "auto"
        self.modo_gordo = False

    def procesar_comando(self, cmd):
        if cmd == "/turbo": 
            self.modo_granularidad = "turbo"
            self.modo_gordo = False
        elif cmd == "/fino": 
            self.modo_granularidad = "fino"
            self.modo_gordo = False
        elif cmd == "/auto": 
            self.modo_carga = "auto"
            self.param_store.liberar_todo()
        elif cmd == "/fijo": 
            self.modo_carga = "fijo"
        elif cmd == "/gordo":
            self.modo_gordo = True
            self.modo_granularidad = "fino"
        print(f"Modo cambiado: Granularidad={self.modo_granularidad}, Carga={self.modo_carga}, Gordo={self.modo_gordo}")

    def forward(self, x, usar_copiloto=True):
        logits_enr = self.enrutador(x)
        logits_p = logits_enr[:, :self.num_departamentos]
        logits_c = logits_enr[:, self.num_departamentos:]
        
        idx_p = torch.argmax(logits_p, dim=1)
        idx_c = torch.argmax(logits_c, dim=1)
        
        h = self.capa_oculta(x, idx_p, idx_c, self.modo_granularidad, usar_copiloto, self.modo_gordo)
        
        # Straight-Through Estimator (STE) para entrenar el enrutador discreto de forma continua
        if self.training:
            probs_p = F.softmax(logits_p, dim=1)
            probs_c = F.softmax(logits_c, dim=1)
            
            w_p_ste = probs_p.gather(1, idx_p.unsqueeze(1))
            w_c_ste = probs_c.gather(1, idx_c.unsqueeze(1))
            
            gate_p = (1.0 + w_p_ste - w_p_ste.detach()).squeeze(1)
            gate_c = (1.0 + w_c_ste - w_c_ste.detach()).squeeze(1)
            
            h = h * gate_p.unsqueeze(1) * gate_c.unsqueeze(1)
            
        return self.fc_salida(h) + self.residual(x), (idx_p, idx_c)

    def save_modelo(self, ruta="modelo_departamental", empaquetar_ligero=True):
        """
        Guarda el modelo. Si empaquetar_ligero=True, purga los tensores gigantes 
        FP32 de base.pt para ahorrar el ~80% del peso en disco.
        """
        if not os.path.exists(ruta): os.makedirs(ruta)
        
        # 1. Exportar y cuantizar obligatoriamente los departamentos independientes a .dpn
        print("\n[Exportación] Cuantizando departamentos a archivos independientes .dpn...")
        for d in range(self.num_departamentos):
            params = self.param_store.pesos_deps[d]
            self._guardar_dpn(d, params)
            
        # 2. Purgar u optimizar el diccionario de pesos maestros
        diccionario_pesos_deps = self.param_store.pesos_deps.state_dict()
        
        if empaquetar_ligero:
            print("[Limpieza] Purgando pesos maestros FP32 del archivo esqueleto...")
            # Reemplazamos las matrices gigantes por referencias estructurales vacías
            diccionario_pesos_deps = {k: torch.empty(0) for k in diccionario_pesos_deps.keys()}
            nombre_archivo = "base_ligero.pt"
        else:
            nombre_archivo = "base.pt"
            
        # 3. Guardar el esqueleto estructural limpio
        torch.save({
            "enrutador": self.enrutador.state_dict(),
            "capa_oculta": {
                "w_p_f": self.capa_oculta.w_p_f, "w_c_f": self.capa_oculta.w_c_f, 
                "bias_f": self.capa_oculta.bias_f, "mascara": self.capa_oculta.mascara_copiloto,
                "fusion_pesada": self.capa_oculta.fusion_pesada.state_dict()
            },
            "fc_salida": self.fc_salida.state_dict(),
            "residual": self.residual.state_dict(),
            "pesos_deps": diccionario_pesos_deps,
            "config": {
                "ni": self.dim_entrada, "nh": self.num_neuronas, 
                "nd": self.num_departamentos, "no": self.fc_salida.out_features,
                "es_ligero": empaquetar_ligero
            }
        }, os.path.join(ruta, nombre_archivo))
        
        print(f"[Éxito] Modelo de producción guardado en: {ruta}/{nombre_archivo}")

    def _guardar_dpn(self, id_dep, params):
        ruta_base = os.path.join(self.param_store.ruta_base)
        if not os.path.exists(ruta_base): os.makedirs(ruta_base)
        nombres = ["mates", "lengua", "fisica", "logica", "historia"]
        ruta = os.path.join(ruta_base, f"{nombres[id_dep]}.dpn")
        
        p_min, p_max = params.min(), params.max()
        scale = (p_max - p_min) / 255.0 if (p_max - p_min) > 0 else 1.0
        zero_point = 128 - p_max / scale
        q_data = ((params / scale) + zero_point).clamp(0, 255).to(torch.uint8)
        
        estado = {"capa_oculta": {"data": q_data, "scale": scale, "zero_point": zero_point}}
        buffer = io.BytesIO()
        torch.save(estado, buffer)
        buffer.seek(0)
        with zipfile.ZipFile(ruta, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.pt", buffer.read())

    @classmethod
    def load_modelo(cls, ruta="modelo_departamental", device='cpu', usar_version_ligera=True):
        """Carga la estructura base adaptándose automáticamente al tipo de empaquetado."""
        nombre_archivo = "base_ligero.pt" if usar_version_ligera else "base.pt"
        ruta_completa = os.path.join(ruta, nombre_archivo)
        
        if not os.path.exists(ruta_completa):
            raise FileNotFoundError(f"No existe el archivo de esqueleto estructural: {ruta_completa}")
            
        base = torch.load(ruta_completa, map_location=device, weights_only=False)
        cfg = base["config"]
        
        m = cls(cfg["ni"], cfg["nh"], cfg["nd"], cfg["no"]).to(device)
        m.enrutador.load_state_dict(base["enrutador"])
        m.capa_oculta.w_p_f.data = base["capa_oculta"]["w_p_f"].to(device)
        m.capa_oculta.w_c_f.data = base["capa_oculta"]["w_c_f"].to(device)
        m.capa_oculta.bias_f.data = base["capa_oculta"]["bias_f"].to(device)
        m.capa_oculta.mascara_copiloto.data = base["capa_oculta"]["mascara"].to(device)
        m.capa_oculta.fusion_pesada.load_state_dict(base["capa_oculta"]["fusion_pesada"])
        m.fc_salida.load_state_dict(base["fc_salida"])
        m.residual.load_state_dict(base["residual"])
        
        # Si no es la versión ligera, se cargan las matrices FP32 redundantes para re-entrenamiento
        if not cfg.get("es_ligero", False):
            m.param_store.pesos_deps.load_state_dict(base["pesos_deps"])
            
        return m

# ---------------------------------------------------------------------------
# 6. CICLO DE ENTRENAMIENTO COMPLETO (Sincronizado con moe_normal.py)
# ---------------------------------------------------------------------------
def entrenar_modelo_departamental(modelo, dataloader, device, epochs=20):
    print(f"\nIniciando Entrenamiento Cooperativo Completo en {device}...")
    modelo.train()  # Habilita el flujo directo de gradientes en el ParamStore
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correctas = 0
        total_muestras = 0
        
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits, _ = modelo(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correctas += (preds == y_batch).sum().item()
            total_muestras += y_batch.size(0)
            
        acc = (correctas / total_muestras) * 100
        print(f"Época {epoch+1:02d}/{epochs} -> Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")

# ---------------------------------------------------------------------------
# 7. VERIFICACIÓN DEL ENTORNO DE EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cpu")
    
    # 1. Instanciar Modelo Completo
    modelo = ModeloDepartamental(dim_entrada=64, num_neuronas=128, num_departamentos=5, dim_salida=10).to(device)
    
    # 2. Simular exactamente la unificación de datos del contrincante clásico
    print("\nGenerando Tensores Cooperativos de Simulación...")
    X_sintetico = torch.randn(1000, 64)
    y_sintetico = torch.randint(0, 10, (1000,))
    loader = DataLoader(TensorDataset(X_sintetico, y_sintetico), batch_size=32, shuffle=True)
    
    # 3. Correr los mismos ciclos exactos de convergencia
    entrenar_modelo_departamental(modelo, loader, device, epochs=20)
    
    # 4. Guardar aplicando Poda de Producción (Activa el empaquetado ligero)
    modelo.save_modelo(ruta="modelo_departamental", empaquetar_ligero=True)
    
    # Borrado físico preventivo del artefacto pesado de laboratorio si existiese
    archivo_antiguo = "modelo_departamental/base.pt"
    if os.path.exists(archivo_antiguo):
        os.remove(archivo_antiguo)
        print("[Limpieza] Archivo redundante base.pt eliminado de forma segura.")
        
    # 5. Cargar la estructura optimizada para Inferencia Limpia
    print("\nCargando modelo purgado de producción para validación...")
    modelo_ligero = ModeloDepartamental.load_modelo("modelo_departamental", device=device, usar_version_ligera=True)
    modelo_ligero.eval()  # Activa la caché LRU e INT8 de forma estricta
    
    # Probar comandos de rendimiento en caliente
    modelo_ligero.procesar_comando("/turbo")
    salida, (p_idx, c_idx) = modelo_ligero(X_sintetico[:4])
    print(f"Inferencia Exitosa! Formato de salida: {salida.shape}")
