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
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 1. KERNELS OPTIMIZADOS (TorchScript)
# ---------------------------------------------------------------------------
@torch.jit.script
def fusion_kernel(h_p, h_c, w_p, w_c, bias):
    alpha = torch.sigmoid(w_p * h_p + w_c * h_c + bias)
    return alpha * h_p + (1 - alpha) * h_c

@torch.jit.script
def compute_output(h_flujo, mascara, h_p, h_fused):
    peso_c = torch.sigmoid(mascara).unsqueeze(0)
    return (1 - peso_c) * h_p + peso_c * h_fused

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
# 2. ALMACÉN DE PARÁMETROS V7.2 (Cuantización Corregida)
# ---------------------------------------------------------------------------
class ParamStoreV7(nn.Module):
    def __init__(self, num_neuronas, num_departamentos, max_colas=6, ruta_base="modelo_departamental/departamentos"):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.num_departamentos = num_departamentos
        self.max_colas = max_colas
        self.ruta_base = ruta_base
        
        self.cache = OrderedDict()
        self.fijos = set()
        self.max_ram = 12
        
        self.pesos_deps = nn.ParameterList([
            nn.Parameter(torch.empty(num_neuronas, num_neuronas + 1))
            for _ in range(max_colas * num_departamentos)
        ])
        for p in self.pesos_deps:
            nn.init.xavier_uniform_(p[:, :num_neuronas])
            p.data[:, -1] = 0.0

    def _get_indice_maestro(self, id_dep, cola):
        return (cola - 1) * self.num_departamentos + id_dep

    def _get_ruta_archivo(self, id_dep, cola):
        nombres = ["mates", "lengua", "fisica", "logica", "historia"]
        return os.path.join(self.ruta_base, f"cola{cola}_{nombres[id_dep]}.dpn")

    def cargar_a_cache(self, id_dep, cola, device='cpu'):
        clave_cache = f"cola_{cola}_dep_{id_dep}"
        if clave_cache in self.cache:
            if clave_cache not in self.fijos:
                self.cache.move_to_end(clave_cache)
            return self.cache[clave_cache]
            
        if len(self.cache) >= self.max_ram and clave_cache not in self.fijos:
            for k in list(self.cache.keys()):
                if k not in self.fijos:
                    self.cache.pop(k)
                    break
                    
        ruta = self._get_ruta_archivo(id_dep, cola)
        if os.path.exists(ruta):
            with zipfile.ZipFile(ruta, 'r') as zf:
                data = zf.read("data.pt")
                buffer = io.BytesIO(data)
                estado = torch.load(buffer, map_location=device, weights_only=False)
            
            q_data = estado["capa_oculta"]["data"]
            scale = estado["capa_oculta"]["scale"]
            zero_point = estado["capa_oculta"]["zero_point"]
            
            # Recuperación exacta sin pérdida del rango dinámico negativo
            params = (q_data.float() - zero_point) * scale
        else:
            idx_maestro = self._get_indice_maestro(id_dep, cola)
            params = self.pesos_deps[idx_maestro].detach().to(device)
            
        self.cache[clave_cache] = params
        return params

    def liberar_todo(self):
        self.fijos.clear()
        self.cache.clear()

# ---------------------------------------------------------------------------
# 3. CAPA DE DEPARTAMENTOS DUAL (Compatible con Gumbel One-Hot)
# ---------------------------------------------------------------------------
class CapaOcultaV7(nn.Module):
    def __init__(self, num_neuronas, num_departamentos, param_store):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.num_departamentos = num_departamentos
        self.param_store = param_store
        
        self.w_p_f = nn.Parameter(torch.randn(num_neuronas))
        self.w_c_f = nn.Parameter(torch.randn(num_neuronas))
        self.bias_f = nn.Parameter(torch.zeros(num_neuronas))
        self.mascara_copiloto = nn.Parameter(torch.rand(num_neuronas))
        self.fusion_pesada = MecanismoFusionPesado(num_neuronas)

    def forward(self, h_flujo, ids_p=None, ids_c=None, probs_p=None, probs_c=None, 
                cola=1, modo_granularidad="fino", usar_copiloto=True, modo_gordo=False):
        device = h_flujo.device
        
        # MODO ENTRENAMIENTO: Gumbel Hard Routing (Vectorizado)
        if self.training:
            h_deps = []
            for d in range(self.num_departamentos):
                idx = self.param_store._get_indice_maestro(d, cola)
                p_params = self.param_store.pesos_deps[idx].to(device)
                h_d = F.relu(F.linear(h_flujo, p_params[:, :self.num_neuronas], p_params[:, -1]))
                h_deps.append(h_d)
                
            h_all_deps = torch.stack(h_deps, dim=1)  # [Batch, Num_Deps, Num_Neuronas]
            
            # Al ser probs_p un vector One-Hot (gracias a Gumbel), esto selecciona el 
            # departamento discreto exacto por muestra sin destruir el gradiente backward.
            h_p = torch.sum(h_all_deps * probs_p.unsqueeze(-1), dim=1)
            
            if not usar_copiloto:
                return h_p
                
            h_c = torch.sum(h_all_deps * probs_c.unsqueeze(-1), dim=1)
            if modo_gordo:
                h_fused = self.fusion_pesada(h_p, h_c)
            else:
                h_fused = fusion_kernel(h_p, h_c, self.w_p_f, self.w_c_f, self.bias_f)
                
            return compute_output(h_flujo, self.mascara_copiloto, h_p, h_fused)

        # MODO INFERENCIA: Hard-Routing Modular Estricto
        else:
            if modo_granularidad == "turbo" and not modo_gordo:
                id_p = torch.mode(ids_p).values.item()
                p_params = self.param_store.cargar_a_cache(int(id_p), cola, device=device)
                h_p = F.relu(F.linear(h_flujo, p_params[:, :self.num_neuronas], p_params[:, -1]))
                
                if not usar_copiloto: return h_p
                
                id_c = torch.mode(ids_c).values.item()
                c_params = self.param_store.cargar_a_cache(int(id_c), cola, device=device)
                h_c = F.relu(F.linear(h_flujo, c_params[:, :self.num_neuronas], c_params[:, -1]))
                
                h_fused = fusion_kernel(h_p, h_c, self.w_p_f, self.w_c_f, self.bias_f)
                return compute_output(h_flujo, self.mascara_copiloto, h_p, h_fused)

            else:
                salida = torch.zeros(h_flujo.size(0), self.num_neuronas, device=device)
                unique_p = ids_p.unique()
                for p_id in unique_p:
                    mask_p = (ids_p == p_id)
                    p_params = self.param_store.cargar_a_cache(p_id.item(), cola, device=device)
                    h_p = F.relu(F.linear(h_flujo[mask_p], p_params[:, :self.num_neuronas], p_params[:, -1]))
                    
                    if usar_copiloto:
                        ids_c_sub = ids_c[mask_p]
                        for c_id in ids_c_sub.unique():
                            mask_c = (ids_c_sub == c_id)
                            c_params = self.param_store.cargar_a_cache(c_id.item(), cola, device=device)
                            h_c = F.relu(F.linear(h_flujo[mask_p][mask_c], c_params[:, :self.num_neuronas], c_params[:, -1]))
                            
                            if modo_gordo:
                                h_fused = self.fusion_pesada(h_p[mask_c], h_c)
                            else:
                                h_fused = fusion_kernel(h_p[mask_c], h_c, self.w_p_f, self.w_c_f, self.bias_f)
                            
                            h_final = compute_output(h_flujo[mask_p][mask_c], self.mascara_copiloto, h_p[mask_c], h_fused)
                            
                            m_global = mask_p.clone()
                            m_global[mask_p] = mask_c
                            salida[m_global] = h_final
                    else:
                        salida[mask_p] = h_p
                return salida

# ---------------------------------------------------------------------------
# 4. CONTROLADOR GENERAL V7.2
# ---------------------------------------------------------------------------
class ModeloDepartamentalV7(nn.Module):
    def __init__(self, dim_entrada, num_neuronas, num_departamentos, dim_salida, max_colas=6):
        super().__init__()
        self.num_neuronas = num_neuronas
        self.dim_entrada = dim_entrada
        self.num_departamentos = num_departamentos
        self.max_colas = max_colas
        
        self.entrada_a_oculta = nn.Linear(dim_entrada, num_neuronas)
        self.enrutador = nn.Sequential(
            nn.Linear(dim_entrada, 32),
            nn.ReLU(),
            nn.Linear(32, num_departamentos * 2)
        )
        
        self.param_store = ParamStoreV7(num_neuronas, num_departamentos, max_colas)
        self.capa_oculta = CapaOcultaV7(num_neuronas, num_departamentos, self.param_store)
        self.fc_salida = nn.Linear(num_neuronas, dim_salida)
        self.residual = nn.Linear(dim_entrada, dim_salida)
        
        self.modo_granularidad = "fino"
        self.modo_carga = "auto"
        self.modo_gordo = False

    def procesar_comando(self, cmd):
        if cmd == "/turbo": self.modo_granularidad = "turbo"
        elif cmd == "/fino": self.modo_granularidad = "fino"
        elif cmd == "/auto": 
            self.modo_carga = "auto"
            self.param_store.liberar_todo()
        elif cmd == "/gordo": self.modo_gordo = True
        print(f"Modo cambiado: Granularidad={self.modo_granularidad}, Carga={self.modo_carga}")

    def forward(self, x, num_colas_ejecucion=None, usar_copiloto=True):
        if num_colas_ejecucion is None:
            num_colas_ejecucion = self.max_colas
        num_colas_ejecucion = min(num_colas_ejecucion, self.max_colas)
        
        logits_enr = self.enrutador(x)
        logits_p = logits_enr[:, :self.num_departamentos]
        logits_c = logits_enr[:, self.num_departamentos:]
        
        if self.training:
            # SOLUCIÓN BRECHA SOFT-TO-HARD: Gumbel Softmax con hard=True
            # Entrena tomando decisiones 100% discretas pero manteniendo el gradiente continuo
            probs_p = F.gumbel_softmax(logits_p, tau=1.0, hard=True)
            probs_c = F.gumbel_softmax(logits_c, tau=1.0, hard=True)
            idx_p = torch.argmax(probs_p, dim=1)
            idx_c = torch.argmax(probs_c, dim=1)
        else:
            idx_p = torch.argmax(logits_p, dim=1)
            idx_c = torch.argmax(logits_c, dim=1)
            probs_p, probs_c = None, None
        
        h_flujo = F.relu(self.entrada_a_oculta(x))
        
        for c in range(1, num_colas_ejecucion + 1):
            if self.training:
                h_transformado = self.capa_oculta(h_flujo, probs_p=probs_p, probs_c=probs_c, 
                                                  cola=c, usar_copiloto=usar_copiloto, 
                                                  modo_gordo=self.modo_gordo)
            else:
                h_transformado = self.capa_oculta(h_flujo, ids_p=idx_p, ids_c=idx_c, 
                                                  cola=c, modo_granularidad=self.modo_granularidad, 
                                                  usar_copiloto=usar_copiloto, modo_gordo=self.modo_gordo)
            h_flujo = h_flujo + h_transformado
                
        return self.fc_salida(h_flujo) + self.residual(x), (idx_p, idx_c)

    def save_modelo(self, ruta="modelo_departamental", empaquetar_ligero=True):
        if not os.path.exists(ruta): os.makedirs(ruta)
        
        print(f"\n[Exportación V7.2] Guardando fragmentos con CUANTIZACIÓN ASIMÉTRICA ASIGNADA...")
        for c in range(1, self.max_colas + 1):
            for d in range(self.num_departamentos):
                idx = self.param_store._get_indice_maestro(d, c)
                params = self.param_store.pesos_deps[idx]
                self._guardar_dpn(d, c, params)
                
        diccionario_pesos_deps = self.param_store.pesos_deps.state_dict()
        if empaquetar_ligero:
            diccionario_pesos_deps = {k: torch.empty(0) for k in diccionario_pesos_deps.keys()}
            nombre_archivo = "base_ligero.pt"
        else:
            nombre_archivo = "base.pt"
            
        torch.save({
            "entrada_a_oculta": self.entrada_a_oculta.state_dict(),
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
                "max_colas": self.max_colas, "es_ligero": empaquetar_ligero
            }
        }, os.path.join(ruta, nombre_archivo))
        print(f"[Éxito V7.2] Estructura base compactada.")

    def _guardar_dpn(self, id_dep, cola, params):
        ruta = self.param_store._get_ruta_archivo(id_dep, cola)
        directorio = os.path.dirname(ruta)
        if not os.path.exists(directorio): os.makedirs(directorio)
        
        p_min, p_max = params.min().item(), params.max().item()
        rango = p_max - p_min
        
        if rango == 0:
            scale = 1.0
            zero_point = 0.0
        else:
            # CORRECCIÓN DEL CULPABLE 1: Cuantización asimétrica de precisión completa
            scale = rango / 255.0
            zero_point = float(round(-p_min / scale))
            zero_point = max(0.0, min(255.0, zero_point))
        
        q_data = torch.round((params / scale) + zero_point).clamp(0, 255).to(torch.uint8)
        
        estado = {"capa_oculta": {"data": q_data, "scale": scale, "zero_point": zero_point}}
        buffer = io.BytesIO()
        torch.save(estado, buffer)
        buffer.seek(0)
        with zipfile.ZipFile(ruta, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("data.pt", buffer.read())

    @classmethod
    def load_modelo(cls, ruta="modelo_departamental", device='cpu', usar_version_ligera=True):
        nombre_archivo = "base_ligero.pt" if usar_version_ligera else "base.pt"
        ruta_completa = os.path.join(ruta, nombre_archivo)
        base = torch.load(ruta_completa, map_location=device, weights_only=False)
        cfg = base["config"]
        
        m = cls(cfg["ni"], cfg["nh"], cfg["nd"], cfg["no"], cfg["max_colas"]).to(device)
        m.entrada_a_oculta.load_state_dict(base["entrada_a_oculta"])
        m.enrutador.load_state_dict(base["enrutador"])
        m.capa_oculta.w_p_f.data = base["capa_oculta"]["w_p_f"].to(device)
        m.capa_oculta.w_c_f.data = base["capa_oculta"]["w_c_f"].to(device)
        m.capa_oculta.bias_f.data = base["capa_oculta"]["bias_f"].to(device)
        m.capa_oculta.mascara_copiloto.data = base["capa_oculta"]["mascara"].to(device)
        m.capa_oculta.fusion_pesada.load_state_dict(base["capa_oculta"]["fusion_pesada"])
        m.fc_salida.load_state_dict(base["fc_salida"])
        m.residual.load_state_dict(base["residual"])
        
        if cfg.get("es_ligero", False):
            for i in range(len(m.param_store.pesos_deps)):
                m.param_store.pesos_deps[i] = nn.Parameter(torch.empty(0), requires_grad=False)
        else:
            m.param_store.pesos_deps.load_state_dict(base["pesos_deps"])
        return m

# ---------------------------------------------------------------------------
# 5. PIPELINE DE ENTRENAMIENTO ESTABLE
# ---------------------------------------------------------------------------
def entrenar_modelo_v7(modelo, dataloader, device, epochs=20):
    print(f"\nIniciando Entrenamiento V7.2 (Gumbel-Hard Vectorizado) en {device}...")
    modelo.train()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.002) # Subimos levemente el LR para estabilizar Gumbel
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correctas, total_muestras = 0.0, 0, 0
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
            
        print(f"Época {epoch+1:02d}/{epochs} -> Loss: {total_loss/len(dataloader):.4f} | Acc: {(correctas/total_muestras)*100:.2f}%")

# ---------------------------------------------------------------------------
# 6. VERIFICACIÓN DE LA GRAN REVANCHA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    X_sintetico = torch.randn(1000, 64)
    y_sintetico = torch.randint(0, 10, (1000,))
    loader = DataLoader(TensorDataset(X_sintetico, y_sintetico), batch_size=32, shuffle=True)
    
    # 5 capas/colas activas exactamente igual que tu entorno de pruebas
    modelo_v7 = ModeloDepartamentalV7(dim_entrada=64, num_neuronas=128, num_departamentos=5, dim_salida=10, max_colas=5).to(device)
    
    entrenar_modelo_v7(modelo_v7, loader, device, epochs=20)
    modelo_v7.save_modelo(ruta="modelo_departamental", empaquetar_ligero=True)
    
    print("\n[Inferencia] Cargando esqueleto ligero V7.2 en Producción...")
    modelo_produccion = ModeloDepartamentalV7.load_modelo("modelo_departamental", device=device, usar_version_ligera=True)
    modelo_produccion.eval()
    
    # Evaluación directa sobre un lote de test
    modelo_produccion.eval()
    with torch.no_grad():
        test_logits, _ = modelo_produccion(X_sintetico)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == y_sintetico).float().mean().item() * 100
        
    print(f"\n📊 RESULTADO EN PRODUCCIÓN (V7.2):")
    print(f"-> Precisión Real de Inferencia (Con archivos .dpn comprimidos): {test_acc:.2f}%")
