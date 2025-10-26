import os
import logging
from typing import List, Optional
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUManager:
    """Gerencia configuração e seleção de GPU para TensorFlow"""
    
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.selected_gpu = None
    
    def _detect_gpus(self) -> List[dict]:
        """Detecta GPUs disponíveis no sistema"""
        gpus = []
        
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            
            for idx, device in enumerate(physical_devices):
                gpu_info = {
                    'id': idx,
                    'name': device.name,
                    'device_type': device.device_type,
                    'available': True
                }
                
                # Tentar obter informações adicionais
                try:
                    details = tf.config.experimental.get_device_details(device)
                    gpu_info['compute_capability'] = details.get('compute_capability')
                except:
                    pass
                
                gpus.append(gpu_info)
            
            logger.info(f"✅ {len(gpus)} GPU(s) detectada(s)")
            
        except Exception as e:
            logger.warning(f"⚠️  Erro ao detectar GPUs: {e}")
        
        return gpus
    
    def list_gpus(self) -> List[dict]:
        """Lista GPUs disponíveis"""
        return self.available_gpus
    
    def select_gpu(self, gpu_id: Optional[int] = None) -> bool:
        """
        Seleciona GPU para uso
        
        Args:
            gpu_id: ID da GPU (None = usar todas, -1 = CPU apenas)
            
        Returns:
            True se configurado com sucesso
        """
        try:
            if gpu_id == -1:
                # Forçar CPU
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                tf.config.set_visible_devices([], 'GPU')
                logger.info("🖥️  CPU mode: GPU desabilitada")
                self.selected_gpu = -1
                return True
            
            if gpu_id is None:
                # Usar todas GPUs
                logger.info("🎮 Usando todas as GPUs disponíveis")
                self.selected_gpu = None
                return True
            
            if gpu_id >= len(self.available_gpus):
                logger.error(f"❌ GPU {gpu_id} não existe")
                return False
            
            # Configurar GPU específica
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.set_visible_devices(physical_devices[0], 'GPU')
                logger.info(f"🎮 GPU {gpu_id} selecionada: {self.available_gpus[gpu_id]['name']}")
                self.selected_gpu = gpu_id
                return True
            else:
                logger.error(f"❌ Não foi possível configurar GPU {gpu_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro ao configurar GPU: {e}")
            return False
    
    def configure_memory_growth(self, enable: bool = True) -> bool:
        """
        Configura crescimento dinâmico de memória GPU
        
        Args:
            enable: True para crescimento dinâmico
            
        Returns:
            True se configurado com sucesso
        """
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if not physical_devices:
                logger.warning("⚠️  Nenhuma GPU para configurar")
                return False
            
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable)
            
            status = "habilitado" if enable else "desabilitado"
            logger.info(f"✅ Crescimento dinâmico de memória {status}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao configurar memória: {e}")
            return False
    
    def set_memory_limit(self, gpu_id: int, memory_mb: int) -> bool:
        """
        Define limite de memória para GPU
        
        Args:
            gpu_id: ID da GPU
            memory_mb: Limite em MB
            
        Returns:
            True se configurado com sucesso
        """
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            
            if gpu_id >= len(physical_devices):
                logger.error(f"❌ GPU {gpu_id} não existe")
                return False
            
            tf.config.set_logical_device_configuration(
                physical_devices[gpu_id],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_mb)]
            )
            
            logger.info(f"✅ Limite de memória GPU {gpu_id}: {memory_mb}MB")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao definir limite de memória: {e}")
            return False
    
    def print_gpu_info(self):
        """Exibe informações das GPUs disponíveis"""
        if not self.available_gpus:
            print("\n❌ Nenhuma GPU detectada")
            print("💡 Verifique:")
            print("   - Drivers NVIDIA instalados")
            print("   - CUDA Toolkit instalado")
            print("   - TensorFlow compatível com sua versão CUDA")
            return
        
        print("\n" + "=" * 60)
        print("🎮 GPUs DISPONÍVEIS")
        print("=" * 60)
        
        for gpu in self.available_gpus:
            print(f"\n[{gpu['id']}] {gpu['name']}")
            print(f"    Tipo: {gpu['device_type']}")
            if 'compute_capability' in gpu:
                print(f"    Compute Capability: {gpu['compute_capability']}")
        
        print("\n" + "=" * 60)
        print("💡 Como selecionar:")
        print("   config.py → GPU_ID = 0, 1, ... (específica)")
        print("   config.py → GPU_ID = None (todas)")
        print("   config.py → GPU_ID = -1 (CPU apenas)")
        print("=" * 60 + "\n")
    
    def get_current_device(self) -> str:
        """Retorna dispositivo atualmente em uso"""
        if self.selected_gpu == -1:
            return "CPU"
        elif self.selected_gpu is None:
            return f"Todas as GPUs ({len(self.available_gpus)})"
        else:
            return f"GPU {self.selected_gpu}"


def setup_gpu(gpu_id: Optional[int] = None, 
              memory_growth: bool = True,
              memory_limit_mb: Optional[int] = None) -> GPUManager:
    """
    Configuração rápida de GPU
    
    Args:
        gpu_id: ID da GPU (None=todas, -1=CPU)
        memory_growth: Habilitar crescimento dinâmico
        memory_limit_mb: Limite de memória em MB
        
    Returns:
        GPUManager configurado
    """
    manager = GPUManager()
    
    # Selecionar GPU
    manager.select_gpu(gpu_id)
    
    # Configurar memória
    if gpu_id != -1:
        if memory_growth:
            manager.configure_memory_growth(True)
        
        if memory_limit_mb and gpu_id is not None and gpu_id >= 0:
            manager.set_memory_limit(gpu_id, memory_limit_mb)
    
    return manager
